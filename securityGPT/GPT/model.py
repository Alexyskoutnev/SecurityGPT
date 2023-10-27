import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters
batch_size = 64
n_embed = 32
block_size = 256 # what is the maximum context length for predictions?
n_head = 6
dropout = 0.2

class Head(nn.Module):

    def __init__(self, head_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 4 x 32 x 4 B 
        B, T, C = x.shape
        k = self.key(x) # 
        q = self.query(x)
        #Computing self attention
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


if __name__ == "__main__":
    pass