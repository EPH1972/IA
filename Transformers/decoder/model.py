import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def load_data(path):
    df = pd.read_parquet(path)
    return df


def initialize_tokenizer(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def create_pairs(words, max_pairs=None):
    X_list = []
    Y_list = []
    
    for i in range(0, len(words) - 5, 6):
        if max_pairs and len(X_list) >= max_pairs:
            break
        X = ' '.join(words[i:i+5])
        Y = words[i+5]
        X_list.append(X)
        Y_list.append(Y)
    
    return X_list, Y_list


def tokenize_data(tokenizer, X_list, Y_list):
    X_tokens = tokenizer(X_list, padding=True, truncation=True, return_tensors="pt")
    Y_tokens = tokenizer(Y_list, padding=True, truncation=True, return_tensors="pt")
    
    X_matrix = X_tokens['input_ids'].numpy()
    # For Y, extract only the first token ID (the word itself) - shape becomes (batch_size,)
    Y_matrix = Y_tokens['input_ids'][:, 0].numpy()
    
    return X_matrix, Y_matrix, X_tokens, Y_tokens


class TransformerDecoder(nn.Module):
    """Transformer Decoder model based on nanoGPT architecture."""
    
    def __init__(self, vocab_size, block_size=512, embedding_dim=768, num_heads=12, num_layers=12, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        
        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(block_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output head
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, X, Y=None):
        B, T = X.shape
        
        # Token embeddings
        token_emb = self.token_embed(X)
        
        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=X.device)
        pos_emb = self.pos_embed(pos)
        
        # Combined embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=X.device) * float('-inf'), diagonal=1)
        
        # Decoder forward pass
        x = self.decoder(x, memory=x, tgt_mask=causal_mask)
        
        # Output logits
        logits = self.lm_head(x)
        
        loss = None
        if Y is not None:
            # Use only the last position logits for prediction
            last_logits = logits[:, -1, :]  # (B, vocab_size)
            loss = F.cross_entropy(last_logits, Y)
        
        return logits, loss


def main():
    # Load data
    df = load_data('/home/iticbcn/IA/Transformers/decoder/train-00025-of-00080.parquet')
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer()
    
    # Combine all text and split into words
    all_text = ' '.join(df['text'].tolist())
    words = all_text.split()
    
    # Create X and Y pairs (limited to 32000)
    X_list, Y_list = create_pairs(words, max_pairs=32000)
    
    # Tokenize data
    X_matrix, Y_matrix, X_tokens, Y_tokens = tokenize_data(tokenizer, X_list, Y_list)
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X_matrix).long()
    Y_tensor = torch.from_numpy(Y_matrix).long()  # Already 1D from tokenize_data
    
    # Create dataset
    dataset = pd.DataFrame({'X': X_list, 'Y': Y_list})
    dataset = dataset.reset_index(drop=True)
    
    # Initialize model
    vocab_size = tokenizer.vocab_size
    model = TransformerDecoder(
        vocab_size=vocab_size,
        block_size=512,
        embedding_dim=768,
        num_heads=12,
        num_layers=6,
        dropout=0.1
    )
    
    # Print model info
    print("Dataset shape:")
    print(f"X: {X_tensor.shape}, Y: {Y_tensor.shape}")
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Forward pass example
    with torch.no_grad():
        logits, loss = model(X_tensor[:10], Y_tensor[:10])
        print(f"\nLogits shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()