import torch.nn as nn
from transformers import AutoModel

class Encoder(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=512):
        super().__init__()
        self.scibert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.proj = nn.Linear(embedding_dim, hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.scibert(input_ids, attention_mask=attention_mask)
        return self.proj(outputs.last_hidden_state)

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, encoder_outputs, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1), 
            encoder_outputs.transpose(0, 1), 
            encoder_outputs.transpose(0, 1)
        )
        return self.fc(attn_out.transpose(0, 1)), hidden