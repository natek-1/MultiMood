import torch
import torch.nn as nn
from torchvision import models

from src.dataset.meld_dataset import MELDDataset


class VideoEncoder(nn.Module):
    '''
    Video encoder module for MELD dataset.

    Args:
        freeze (bool): Whether to freeze the R3D-18 model (default: True).
        hidden_dim (int): Dimension of the output of the text encoder (default: 128).
        dropout_rate (float): Dropout rate (default: 0.2).
    
    '''

    def __init__(self, freeze = True, hidden_dim=128, dropout_rate = 0.2):
        super().__init__()
        weights = models.video.R3D_18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = models.video.r3d_18(weights=weights)
        num_fts = self.model.fc.in_features
        self.dropout_rate = dropout_rate

        if freeze:
            self._freeze()
        
        self.model.fc = nn.Linear(num_fts, hidden_dim) 
        self.post_model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
    
    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, video_tensor: torch.Tensor):
        '''
        Forward pass for the video encoder.

        Args:
            video_tensor (torch.Tensor): Preprocessed video tensor of shape
            (batch, channel, time_stamp, height, width)
        
        Returns:
            torch.Tensor: Output of the video encoder. (batch, hidden_dim)
        '''
        video_tensor = self.model(video_tensor)
        return self.post_model(video_tensor)


if __name__ == "__main__":
    model = VideoEncoder()
    dataset = MELDDataset("meld_dataset/dev/dev_sent_emo.csv",
                          "meld_dataset/dev/dev_splits_complete",
                          preprocess = model.preprocess)
    
    for item in dataset:
        if item is not None:
            video_tensor = item['video_frames'].unsqueeze(0)
            print(video_tensor.shape)
            output = model(video_tensor)
            print(output.shape)

            break
