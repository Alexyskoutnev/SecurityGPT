import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F

from securityGPT.utils.gpt_utils import *

CONFIG_NAME_GPT = "test_gpt.yml"
CONFIG_NAME_DATASET = "dataset.yml"
CONFIG_PATH_GPT = os.path.join("../config", CONFIG_NAME_GPT)
CONFIG_PATH_DATASET = os.path.join("../config", CONFIG_NAME_DATASET)
with open(CONFIG_PATH_GPT, 'r') as cfg_file:
    GPT_config = yaml.safe_load(cfg_file)
with open(CONFIG_PATH_DATASET, 'r') as cfg_file:
    dataset_config = yaml.safe_load(cfg_file)

class Head(nn.Module):

    def __init__(self, n_embed, block_size, head_size, dropout=0.2):
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

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embed, dropout=0.2):
        super.__init__()
        self.heads = nn.ModuleList([Head(head_size, dropout) for i in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FF(nn.Module):
    def __init__(self, n_embd, dropout=0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FF(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embed'])
        self.position_embedding_table = nn.Embedding(config['block_size'], config['n_embed'])
        self.blocks = nn.Sequential(*[Block(config['n_embed'], n_head=config['n_head']) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embed'])
        self.lm_head = nn.Linear(config['n_embed'], config['vocab_size'])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config['device']))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['block_size']:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
        
    model = GPT(config=GPT_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_config['learning_rate'])

    for iter in range(GPT_config['max_iters']):

        # every once in a while evaluate the loss on train and val sets
        if iter % GPT_config['eval_interval'] == 0 or iter == GPT_config['max_iters'] - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
