import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()

        self.attention = SelfAttention(embed_dim=embed_dim, heads=heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Make hidden layer of the feedforward 2 times as big as the input and output
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        feedforward = self.ff(x)
        return self.norm2(feedforward+x)


class Transformer(nn.Module):
    def __init__(self, embed_dim, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, embed_dim)
        self.pos_emb = nn.Embedding(seq_length, embed_dim)

        transformer_blocks = []

        for i in range(depth):
            transformer_blocks.append(TransformerBlock(embed_dim=embed_dim, heads=heads))

        self.tblocks = nn.Sequential(*transformer_blocks)

        self.probs = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        x = self.probs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)

