from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import build_image_transform, load_image
from .tokenizer import SimpleTokenizer, TokenizerConfig


@dataclass
class Sample:
    image_path: str
    text: str


class ImageTextDataset(Dataset):
    def __init__(self, json_path: str, tokenizer: SimpleTokenizer, image_size: int = 224):
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.transform = build_image_transform(image_size=image_size)

        with self.json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = [Sample(**item) for item in raw]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        image_path = sample.image_path
        if not Path(image_path).is_absolute():
            image_path = str((self.json_path.parent / image_path).resolve())

        image = self.transform(load_image(image_path))
        tokens = self.tokenizer.pad(self.tokenizer.encode(sample.text))

        return {
            "image": image,
            "text": torch.tensor(tokens, dtype=torch.long),
            "raw_text": sample.text,
            "image_path": image_path,
        }


def build_tokenizer_from_json(json_path: str, config: TokenizerConfig | None = None) -> SimpleTokenizer:
    with Path(json_path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    texts = [item["text"] for item in raw]
    return SimpleTokenizer.build(texts, config=config)


def collate_batch(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    texts = torch.stack([item["text"] for item in batch], dim=0)

    return {
        "images": images,
        "texts": texts,
        "raw_texts": [item["raw_text"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def build_dataloader(
    json_path: str,
    tokenizer: SimpleTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    image_size: int = 224,
    num_workers: int = 4,
):
    dataset = ImageTextDataset(json_path=json_path, tokenizer=tokenizer, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

