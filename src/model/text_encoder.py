import torch
import torch.nn as nn
from transformers import BertModel

from src.dataset.meld_dataset import MELDDataset

class TextEncoder(nn.Module):
    '''
    Text encoder module for MELD dataset.

    Args:
        freeze (bool): Whether to freeze the BERT model (default: True).
    '''
    BERT_DIM = 768
    
    def __init__(self, freeze=True, hidden_dim = 128):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        if freeze:
            self._freeze_bert()
        self.projection = nn.Linear(self.BERT_DIM, hidden_dim)
    

    def _freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, 
                input_ids : torch.Tensor, 
                attention_mask : torch.Tensor
        ) -> torch.Tensor:
        '''
        Forward pass for the text encoder.

        Args:
            input_ids (torch.Tensor): Input IDs for the BERT model.
            attention_mask (torch.Tensor): Attention mask for the BERT model.
        
        Returns:
            torch.Tensor: Output of the text encoder.
        '''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output : torch.Tensor = outputs.pooler_output
        return torch.nn.functional.relu(self.projection(output))
    

if __name__ == "__main__":
    dataset = MELDDataset("meld_dataset/dev/dev_sent_emo.csv",
                          "meld_dataset/dev/dev_splits_complete")
    model = TextEncoder()
    for item in dataset:
        if item is not None:
            attention_mask = item['attention_mask'].unsqueeze(0)
            input_ids = item['input_ids'].unsqueeze(0)
            output = model(input_ids, attention_mask)
            print(output.shape)

            break