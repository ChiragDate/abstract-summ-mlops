import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .model_arch import Encoder, Decoder
from .model_utils import PaperDataset, load_data
from .modelconfig import config

def train():
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[DOC]', '[ABS]', '[EOS]']})
    
    # Load data
    df = load_data()
    dataset = PaperDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Model
    encoder = Encoder().to(config.DEVICE)
    decoder = Decoder(len(tokenizer)).to(config.DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=config.LEARNING_RATE
    )
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        encoder.train()
        decoder.train()
        epoch_loss = 0
        
        for batch in dataloader:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            # Forward pass
            encoder_outputs = encoder(batch['input_ids'], batch['attention_mask'])
            outputs, _ = decoder(
                batch['labels'][:, :-1],  # Teacher forcing
                encoder_outputs
            )
            
            # Loss calculation
            loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
                outputs.reshape(-1, outputs.shape[-1]),
                batch['labels'][:, 1:].reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 
                1.0
            )
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Save model
    torch.save({
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'tokenizer': tokenizer,
        'config': config.__dict__
    }, "models/abstract_generator.pt")

if __name__ == "__main__":
    train()