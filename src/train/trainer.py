import os
from datetime import datetime
from typing import Literal, Dict

from tqdm import tqdm


from sklearn.metrics import precision_score, accuracy_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from typing import cast


from src.dataset.meld_dataset import MELDDataset
from src.model.sentiment_model import MultiModelSentimentModel
from src.train.utils import compute_class_weight


class MultiModalTrainer:
    def __init__(self, model : MultiModelSentimentModel, train_loader: DataLoader,
                device: Literal['cpu', 'cuda', 'mps'] = "cpu"):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        
        # ex: Jun27_08-22-16
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        base_dir = '/opt/ml/output/tensorbord' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr':5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr':5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr':5e-4}
        ], weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=2
        )

        self.current_train_losses: None | Dict[str, float]  = None

        emotion_weights, sentiment_weights = compute_class_weight(
            cast(MELDDataset, train_loader.dataset)
        )

        self.emotion_weights = emotion_weights.to(device)
        self.sentiment_weights = sentiment_weights.to(device)

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight=self.emotion_weights
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight=self.sentiment_weights
        )


    def log_metrics(self, losses : Dict[str, float], metrics : None | Dict[str, torch.Tensor] = None,
                    phase : Literal['train', 'val', 'test'] = "train", epoch: None | int = None):
        
        step = epoch if epoch else self.global_step
        
        if phase == "train":
            self.current_train_losses = losses
        self.writer.add_scalar(
            f'loss/total/{phase}' , losses['total'], step
        )
        self.writer.add_scalar(
            f'loss/emotion/{phase}', losses['emotion'], step
        )
        self.writer.add_scalar(
            f'loss/sentiment/{phase}', losses['sentiment'], step
        )

        if metrics:
            self.writer.add_scalar(
                f'{phase}/emotion_precision', metrics['emotion_precision'],
                step
            )
            self.writer.add_scalar(
                f'{phase}/emotion_accuracy', metrics['emotion_accuracy'],
                step
            )
            self.writer.add_scalar(
                f'{phase}/emotion_recall', metrics['emotion_recall'],
                step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_precision', metrics['sentiment_precision'],
                step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_accuracy', metrics['sentiment_accuracy'],
                step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_recall', metrics['sentiment_recall'],
                step
            )
    
    def train_epoch(self):
        self.model.train()
        running_loss = {
            'total': 0, 'emotion': 0, 'sentiment': 0
        }
        iterator = tqdm(self.train_loader, total=len(self.train_loader), leave=False,
                    desc="Training")
        for batch in iterator:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            video_frames = batch['video_frames'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)
            sentiment_labels = batch['sentiment_label'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask, 
                                video_frames, audio_features)
        
            emotion_loss = self.emotion_criterion(
                outputs['emotions'], emotion_labels
            )
            sentiment_loss = self.sentiment_criterion(
                outputs['sentiments'], sentiment_labels
            )
            total_loss = emotion_loss + sentiment_loss

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.optimizer.step()

            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
            })
            iterator.set_description(f"Training Loss: {total_loss.item():.4f}")

            self.global_step += 1
        return {key: value/len(self.train_loader) for key, value in running_loss.items()}


    def evaluate(self, dataloader: DataLoader, epoch: int,
                 phase: Literal['val', 'test'] = 'val'):
        ''' runs on test or validation loader
        '''
        self.model.eval()
        running_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in tqdm(dataloader, total=len(dataloader), leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                video_frames = batch['video_frames'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                sentiment_labels = batch['sentiment_label'].to(self.device)

                # make prediction
                outputs = self.model(input_ids, attention_mask, 
                                    video_frames, audio_features)
            
                emotion_loss = self.emotion_criterion(
                    outputs['emotions'], emotion_labels
                )
                sentiment_loss = self.sentiment_criterion(
                    outputs['sentiments'], sentiment_labels
                )
                total_loss = emotion_loss + sentiment_loss

                emotion_predictions = outputs['emotions'].argmax(dim=-1)
                sentiment_prediction = outputs['sentiments'].argmax(dim=-1)
                all_emotion_preds.extend(
                    emotion_predictions.cpu().numpy()
                )
                all_emotion_labels.extend(
                    emotion_labels.cpu().numpy()
                )
                all_sentiment_preds.extend(
                    sentiment_prediction.cpu().numpy()
                )
                all_sentiment_labels.extend(
                    sentiment_labels.cpu().numpy()
                )

                running_loss['total'] += total_loss.item()
                running_loss['emotion'] += emotion_loss.item()
                running_loss['sentiment'] += sentiment_loss.item()
        
        avg_loss = {key: value/len(dataloader) for key, value in running_loss.items()}

        # metrics
        emotion_precision = precision_score(
            all_emotion_labels, all_emotion_preds, average='weighted'
        )
        emotion_recall = recall_score(
            all_emotion_labels, all_emotion_preds, average='weighted'
        )
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        sentiment_precision = precision_score(
            all_sentiment_labels, all_sentiment_preds, average='weighted'
        )
        sentiment_recall = recall_score(
            all_sentiment_labels, all_sentiment_preds, average='weighted'
        )
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)

        print("Emotion precision: ", emotion_precision)
        print("Emotion recall: ", emotion_recall)
        print("Emotion accuracy: ", emotion_accuracy)
        print("Sentiment precision: ", sentiment_precision)
        print("Sentiment recall: ", sentiment_recall)
        print("Sentiment accuracy: ", sentiment_accuracy)

        metrics = {
            'emotion_precision': emotion_precision,
            'emotion_recall': emotion_recall,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_recall': sentiment_recall,
            'sentiment_accuracy': sentiment_accuracy
        }

        self.log_metrics(avg_loss, metrics, phase, epoch)

        if phase == "val":
            self.scheduler.step(avg_loss['total'])
        
        return avg_loss, metrics


if __name__ == "__main__":
    device = "cpu"
    model = MultiModelSentimentModel().to(device)
    dataset = MELDDataset("meld_dataset/dev/dev_sent_emo.csv",
                          "meld_dataset/dev/dev_splits_complete",
                          preprocess=model.preprocess)
    # using dataloader bc batchnorm requires batch size greater than 1
    dataloader = DataLoader(dataset, batch_size=8)
    trainer = MultiModalTrainer(model, dataloader, device)
    trainer.train_epoch()
    trainer.evaluate(dataloader, 0, 'val')
    














