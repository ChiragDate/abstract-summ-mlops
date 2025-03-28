import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import model_utils as ModelConfig

class PaperDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize main paper text
        inputs = self.tokenizer(
            row['main_paper'],
            max_length=ModelConfig.MAX_INPUT_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize abstract
        targets = self.tokenizer(
            row['abstract'],
            max_length=ModelConfig.MAX_TARGET_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def load_data():
    df = pd.read_csv(ModelConfig.CSV_PATH)
    
    # Ensure required columns exist
    assert all(col in df.columns for col in ['main_paper', 'abstract'])
    
    # Add special tokens if not already present
    df['abstract'] = "[ABS] " + df['abstract'] + " [EOS]"
    df['main_paper'] = "[DOC] " + df['main_paper']
    
    return df