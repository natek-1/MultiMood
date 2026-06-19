import os
from typing import Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, BertTokenizer



os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MELDDataset(Dataset):
    '''
    MELD Dataset class.

    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        video_dir (str): Directory containing the video files.
    '''

    def __init__(self, csv_path : str, video_dir : str):
        self.data : pd.DataFrame = pd.read_csv(csv_path)
        self.video_dir : str = video_dir
        self.tokenizer : BertTokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.emotion_map : Dict[str, int] = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }

        self.sentiment_map : Dict[str, int] = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
    
    def __len__(self):
        return len(self.data)

    def _load_video_frames(self, video_path, max_frames = 30, size=(244, 244)) -> torch.Tensor:
        '''
        Load video frames from the given video path.

        Args:
            video_path (str): Path to the video file.
            max_frames (int): Maximum number of frames to extract (default: 30).
            size (tuple): Size of the frames to resize to (default: (244, 244)).
        
        Returns:
            torch.Tensor: Tensor of shape (num_frames, height, width, color_channels).
        '''

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

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        dia_id = row['Dialogue_ID']
        utt_id = row['Utterance_ID'] 
        video_filename = f"dia{dia_id}_utt{utt_id}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"No video found for the path {video_path}"
            )

        text = self.tokenizer(row['Utterance'], padding='max_length',
                              truncation=True, max_length=128, return_tensors='np')
        input_ids: torch.Tensor = torch.tensor(text['input_ids'])
        attention_mask: torch.Tensor = torch.tensor(text['attention_mask'])
        

        video_frames: torch.Tensor = self._load_video_frames(video_path)
        assert video_frames.shape[0] == 30

        


    


if __name__ == "__main__":
    datasets = [("meld_dataset/dev/dev_sent_emo.csv", "meld_dataset/dev/dev_splits_complete"),
                ("meld_dataset/test/test_sent_emo.csv", "meld_dataset/test/output_repeated_splits_test"),
                ("meld_dataset/train/train_sent_emo.csv", "meld_dataset/train/train_splits")]
    csv_path, frames_dir = datasets[2]
    dataset = MELDDataset(csv_path, frames_dir)
    for idx in tqdm(range(1160, len(dataset)), total=len(dataset)-1160, leave=False):
        if idx in (1084, 1165): continue
        dataset[idx]

    

    