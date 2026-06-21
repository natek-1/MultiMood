import torch
import torch.nn as nn


class AudioEncoder(nn.Module):

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
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
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
