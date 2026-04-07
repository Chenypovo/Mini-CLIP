from __future__ import annotations

import torch
from torch import nn

from .image_encoder import SmallImageEncoder
from .text_encoder import SmallTextEncoder

class MiniCLIP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 32,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.image_encoder = SmallImageEncoder(embed_dim = embed_dim)
        self.text_encoder = SmallTextEncoder(
            vocab_size = vocab_size,
            max_length = max_length,
            embed_dim = embed_dim,
            channel = 256,
            heads = 4,
            layers = 2,
            pad_id = 0,
    )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor: 
        return self.image_encoder(images)
    
    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(tokens)
    
    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        image_features = self.encode_image(images) # [B, E]
        text_features = self.encode_text(tokens) # [B, E]

        scale = self.logit_scale.exp()

        # image_features: [B, E]
        # text_features.t(): [E, B]
        # logits_per_image: [B, B]
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text