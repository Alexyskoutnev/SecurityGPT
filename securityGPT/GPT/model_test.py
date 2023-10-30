import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union

from securityGPT.utils.gpt_utils import *
from securityGPT.utils.utils import Cfgloader

class Head(nn.Module):

    def __init__(self, n_embed : int , block_size : int, head_size : int, dropout : float = 0.2) -> None:
        """
        Initialize a single head for multi-head attention.

        Args:
            n_embed (int): The input embedding size.
            block_size (int): The size of the attention block.
            head_size (int): The size of the attention head.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor):
        """
        Forward pass for the single head in multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B is batch size, T is the sequence length, and C is the number of features.

        Returns:
            torch.Tensor: The output tensor after applying attention.
        """
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

    def __init__(self, num_heads : int, head_size : int, n_embed : int, dropout : float = 0.2, block_size : int = 256):
        """
        Initialize a multi-head attention layer.

        Args:
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
            n_embed (int): The input embedding size.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
            block_size (int, optional): The size of the attention block. Defaults to 256.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, block_size, head_size, dropout) for i in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor):
        """
        Forward pass for the multi-head attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B is batch size, T is the sequence length, and C is the number of features.

        Returns:
            torch.Tensor: The output tensor after applying multi-head attention.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FF(nn.Module):
    def __init__(self, n_embd : int, dropout : float = 0.2) -> None:
        """
        Initialize a feedforward neural network layer.

        Args:
            n_embd (int): The input embedding size.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x : torch.Tensor) -> torch.tensor:
        """
        Forward pass for the feedforward neural network layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B is batch size, T is the sequence length, and C is the number of features.

        Returns:
            torch.Tensor: The output tensor after applying the feedforward neural network layer.
        """
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd : int, n_head : int, dropout : float = 0.2, block_size : int = 256) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FF(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config : dict = None) -> None:
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding_table = nn.Embedding(config['block_size'], config['n_embd'])
        self.blocks = nn.Sequential(*[Block(n_embd=config['n_embd'], n_head=config['n_head'], 
                                    dropout=config['dropout'], block_size=config['block_size']) 
                                    for _ in range(config['n_layers'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'])
        self.apply(self._init_weights)

    def _init_weights(self, module : object) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx : torch.tensor, targets : torch.tensor = None) -> Union[torch.tensor, torch.tensor]:
        """
        Forward pass for the GPT model.

        Args:
            idx (torch.Tensor): Input tensor representing the text.
            targets (torch.Tensor): Target tensor for training.

        Returns:
            Union[torch.Tensor, torch.Tensor]: If targets are None, returns logits; otherwise, returns logits and loss.
        """
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

    def generate(self, idx : torch.tensor, max_new_tokens : int) -> torch.tensor:
        """
        Generate new text using the GPT model.

        Args:
            idx (torch.Tensor): Input tensor representing the text.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            torch.Tensor: Generated text.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['block_size']:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    CONFIG_NAME_GPT = "test_gpt.yml"
    CONFIG_NAME_DATASET = "dataset.yml"
    DATASET_PATH = "../../data"
    MODEL_SAVE_DIR = "../../models/gpt"
    cfg_loader = Cfgloader("../config")
    GPT_config = cfg_loader.load_config("test_gpt.yml")
    dataset_config = cfg_loader.load_config("dataset.yml")
    """DATA LOADING (TEMP)
    """
    dataset_path = os.path.join(DATASET_PATH, 'Shakespeare.txt')
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    GPT_config['vocab_size'] = vocab_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(config=GPT_config).to(device)
    GPT_config['device'] = device
    print('Using device:', device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(GPT_config['learning_rate']))

    for iter in range(GPT_config['max_iters']):

        # every once in a while evaluate the loss on train and val sets
        if iter % GPT_config['eval_interval'] == 0 or iter == GPT_config['max_iters'] - 1:
            losses = estimate_loss(model, GPT_config['eval_iters'], train_data, val_data, GPT_config['block_size'], GPT_config['batch_size'], device=GPT_config['device'])
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data, GPT_config['block_size'], GPT_config['batch_size'], device=GPT_config['device'])

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_model(model, MODEL_SAVE_DIR)
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
