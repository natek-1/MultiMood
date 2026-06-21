import os
from pathlib import Path
from typing import Dict, Tuple
import subprocess

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torchaudio
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, BertTokenizer



os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MELDDataset(Dataset):
    """
    MELD Dataset class.

    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        video_dir (str): Directory containing the video files.
        preprocess (callable): Preprocessing function for video frames.
    """

    def __init__(self, csv_path : str, video_dir : str, preprocess=None):
        self.data : pd.DataFrame = pd.read_csv(csv_path)
        self.video_dir : str = video_dir
        self.tokenizer : BertTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.preprocess = preprocess

        self.emotion_map : Dict[str, int] = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }

        self.sentiment_map : Dict[str, int] = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
    
    def __len__(self):
        return len(self.data)

    def _load_video_frames(self, video_path: str, max_frames : int = 30, size : Tuple =(244, 244)) -> torch.Tensor:
        """
        Load video frames from the given video path.

        Args:
            video_path (str): Path to the video file.
            max_frames (int): Maximum number of frames to extract (default: 30).
            size (tuple): Size of the frames to resize to (default: (244, 244)).
        
        Returns:
            torch.Tensor: Tensor of shape (num_frames, height, width, color_channels).
        """

        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found {video_path}")
            
            # Read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
            while len(frames) < max_frames and cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame, size)
                # normalize rgb channel
                frame = frame / 255.0
                frames.append(frame)
        except Exception as e:
            raise ValueError(f"Video Processing Error: {str(e)}")
        finally:
            cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames could be extracted")

        # Pad
        if len(frames) < max_frames:
            frames += [np.zeros_like(frames[0])] * (max_frames - len(frames))

        # shape (num_frames, height, width color_channel)
        frames = np.array(frames)

        # torch processes images as (num_frames, color_channels, height, width)
        return torch.FloatTensor(frames).permute(0, 3, 1, 2)
    
    def _extract_audio_features(self, video_path: str, rate: int = 16000,
                                n_mels: int = 64, n_fft: int = 1024, hop_length: int = 512,
                                normalize: bool =True, max_size : int = 300) -> torch.Tensor:
        '''
        Extract audio features from the given video path.

        Args:
            video_path (str): Path to the video file.
            rate (int): Sample rate of the audio (default: 16000).
            n_mels (int): Number of Mel bands (default: 64).
            n_fft (int): Number of FFT points (default: 1024).
            hop_length (int): Number of hop points (default: 512).
            normalize (bool): Whether to normalize the audio features (default: True).
            max_size (int): Maximum size of the audio features (default: 300).
        
        Returns:
            torch.Tensor: Tensor of shape (n_mels, max_size).
        '''
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-vn", # no video output
                "-acodec", "pcm_s16le",
                "-ar", str(rate), # sample rate
                '-ac', '1', # monoaudio
                audio_path # output path
            ], check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != rate:
                resampler = torchaudio.transforms.Resample(sample_rate, rate)
                waveform = resampler(waveform)
            waveform = waveform.squeeze(0)
            mel_spectogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )
            mel_spec : torch.Tensor = mel_spectogram(waveform)
            if normalize:
                mel_spec = (mel_spec - mel_spec.mean())/mel_spec.std()
            
            if mel_spec.size(1) < max_size:
                padding = max_size - mel_spec.size(1)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :max_size]

            return mel_spec
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio Error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)




    def __getitem__(self, idx: int | torch.Tensor) -> Dict[str, torch.Tensor] | None:
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        row = self.data.iloc[idx]
        try:
            dia_id = row["Dialogue_ID"]
            utt_id = row["Utterance_ID"] 
            video_filename = f"dia{dia_id}_utt{utt_id}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            if not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"No video found for the path {video_path}"
                )

            text = self.tokenizer(row["Utterance"], padding="max_length",
                                truncation=True, max_length=128, return_tensors="np")
            input_ids: torch.Tensor = torch.tensor(text["input_ids"])
            attention_mask: torch.Tensor = torch.tensor(text["attention_mask"])
            

            video_frames = self._load_video_frames(video_path)
            if self.preprocess is not None:
                video_frames : torch.Tensor = self.preprocess(video_frames)
            audio_features = self._extract_audio_features(video_path)
            
            # map sentiment and emotion
            emotion_label = torch.tensor(
                self.emotion_map[row['Emotion'].lower()]
            )
            sentiment_label = torch.tensor(
                self.sentiment_map[row['Sentiment'].lower()]
            )

            return {
                'input_ids': input_ids.squeeze(),
                'attention_mask': attention_mask.squeeze(),
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': emotion_label,
                'sentiment_label': sentiment_label
            }
        except Exception as e:
            print(f"Error processing {idx}: {str(e)}")
            return None


        

def collate_fn(batch):
    batch = list(filter(None, batch))
    return default_collate(batch)


def prepare_dataloader(train_csv, train_video_dir,
                       dev_csv, dev_video_dir,
                       test_csv, test_video_dir, batch_size=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Prepare dataloader for the MELD dataset.
    Args:
        train_csv (str): Path to the training CSV file.
        train_video_dir (str): Directory containing the training video files.
        dev_csv (str): Path to the development CSV file.
        dev_video_dir (str): Directory containing the development video files.
        test_csv (str): Path to the test CSV file.
        test_video_dir (str): Directory containing the test video files.
        batch_size (int): Batch size (default: 32).
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Tuple of dataloaders for training, development, and testing.
    '''
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader



    


if __name__ == "__main__":
    datasets = [
            ("meld_dataset/train/train_sent_emo.csv", "meld_dataset/train/train_splits"),
            ("meld_dataset/dev/dev_sent_emo.csv", "meld_dataset/dev/dev_splits_complete"),
            ("meld_dataset/test/test_sent_emo.csv", "meld_dataset/test/output_repeated_splits_test"),
        ]
    train_loader, dev_loader, test_loader = prepare_dataloader(
        datasets[0][0], datasets[0][1],
        datasets[1][0], datasets[1][1],
        datasets[2][0], datasets[2][1]
    )

    for batch in train_loader:
        print(batch['input_ids'])
        print(batch['attention_mask'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break


    

    