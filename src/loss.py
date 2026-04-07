from __future__ import annotations

import torch
import torch.nn.functional as F

def clip_contrasive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size, device = logits_per_image.device)

    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)

    return (loss_i + loss_t) / 2