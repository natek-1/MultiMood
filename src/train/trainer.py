import os
from datetime import datetime

from tqdm import tqdm

from sklearn.metrics import precision_score, accuracy_score

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter


from src.dataset.meld_dataset import MELDDataset
from src.model.sentiment_model import MultiModelSentimentModel


class MultiModalTrainer:
    def __init__(self, model : nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, device: str = "test"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        

