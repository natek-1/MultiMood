import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    '''
    Audio encoder module for MELD dataset.

    Args:
        hidden_dim (int): Dimension of the output of the text encoder (default: 128).
        dropout_rate (float): Dropout rate (default: 0.2).
    '''

    def __init__(self, hidden_dim = 128, dropout_rate = 0.2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.projection = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the audio encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_mels, time).
        
        Returns:
            torch.Tensor: Output of the audio encoder of shape (batch, hidden_dim).
        '''
        x = self.conv_layers(x)
        return self.projection(x.squeeze(-1))
    

if __name__ == "__main__":
    from src.dataset.meld_dataset import MELDDataset
    model = AudioEncoder()
    dataset = MELDDataset("meld_dataset/dev/dev_sent_emo.csv",
                          "meld_dataset/dev/dev_splits_complete")
    
    for item in dataset:
        if item is not None:
            audio_features = item['audio_features'].unsqueeze(0)
            output = model(audio_features)
            print(output.shape)

            break
