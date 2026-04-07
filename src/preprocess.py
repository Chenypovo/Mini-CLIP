from __future__ import annotations

from PIL import Image
from torchvision import transforms

def build_image_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        ]
    )

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
