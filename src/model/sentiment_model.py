from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.model.audio_encoder import AudioEncoder
from src.model.text_encoder import TextEncoder
from src.model.video_encoder import VideoEncoder

class MultiModelSentimentModel(nn.Module):

    def __init__(self, hidden_dim=128, feat_dim=256, dropout_rate=0.3,
                        num_classification=7, num_sentiments = 3):
        super().__init__()

        self.text_encoder = TextEncoder(hidden_dim = hidden_dim)
        self.video_encoder = VideoEncoder(hidden_dim = hidden_dim)
        self.audio_encoder = AudioEncoder(hidden_dim = hidden_dim)
        self.preprocess = self.video_encoder.preprocess

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(3 * hidden_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classifiers
        self.emotion_classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classification)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_sentiments)
        )
    
    def forward(self, input_ids, attention_mask,
                video_frames, audio_features) -> Dict[str, torch.Tensor]:
        
        # text, video, audio features will have shape [batch_size, hidden_dim]
        text_features = self.text_encoder(
            input_ids,
            attention_mask
        ) 
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=-1) # batch_size, 3*hidden_dim

        # output shape: batch_size, feat_dim
        fused_features = self.fusion_layer(combined_features)

        emotion = self.emotion_classifier(fused_features)
        sentiment = self.sentiment_classifier(fused_features)

        return {
            'emotion': emotion,
            'sentiment': sentiment
        }

if __name__ == "__main__":
    from src.dataset.meld_dataset import MELDDataset
    device = "cpu"
    model = MultiModelSentimentModel().to(device)
    dataset = MELDDataset("meld_dataset/dev/dev_sent_emo.csv",
                          "meld_dataset/dev/dev_splits_complete",
                          preprocess=model.preprocess)
    # using dataloader bc batchnorm requires batch size greater than 1
    dataloader = DataLoader(dataset, batch_size=2) 
    
    for item in dataloader:
        if item is not None:
            audio_features = item['audio_features'].to(device)
            video_tensor = item['video_frames'].to(device)
            attention_mask = item['attention_mask'].to(device)
            input_ids = item['input_ids'].to(device)
            output = model(
                input_ids, attention_mask, video_tensor, audio_features
            )
            emotion = output['emotion']
            sentiment = output['sentiment']
            print(emotion.shape)
            print(sentiment.shape)

            break
        