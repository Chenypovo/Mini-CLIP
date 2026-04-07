from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class SmallTextEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_length: int = 32,
            embed_dim: int = 256,
            channel: int = 512,
            heads: int = 4,
            layers: int = 2,
            pad_id = 0
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, channel) # [B, L] -> [B, L, C]
        self.positional_embedding = nn.Parameter(torch.randn(max_length, channel) * 0.01) # [L, C]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = channel,
            nhead = heads,
            dim_feedforward = channel * 4,
            batch_first = True,
            activation = "gelu",
            dropout = 0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = layers)
        self.ln_final = nn.LayerNorm(channel)
        self.projector = nn.Linear(channel, embed_dim) # [B, C] -> [B, E]
        self.max_length = max_length


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L]
        mask = tokens != self.pad_id # mask: [B, L], True 表示有效 token，False 表示 pad

        x = self.token_embedding(tokens) # x: [B, L, C]
        # positional_embedding[:L]: [L, C]
        # unsqueeze(0): [1, L, C]
        # x: [B, L, C]
        x = x + self.positional_embedding[: x.shape[1], :].unsqueeze(0)
        x = self.transformer(x)
        x = self.ln_final(x)

        mask_f = mask.unsqueeze(-1).float() # mask_f: [B, L, 1]
        # x * mask_f: [B, L, C]
        # summed: [B, C]
        summed = (x * mask_f).sum(dim = 1)
        denom = mask_f.sum(dim = 1).clamp(min = 1.0) # denom: [B, 1]
        pooled = summed / denom

        features = self.projector(pooled)
        features = F.normalize(features, dim = -1)
        return features
