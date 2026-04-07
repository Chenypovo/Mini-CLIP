from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.model import MiniCLIP
from src.preprocess import build_image_transform
from src.tokenizer import SimpleTokenizer, TokenizerConfig


def resolve_image_path(image_path: str, data_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return (Path(data_path).parent / p).resolve()


def is_valid_image(path: Path) -> bool:
    try:
        Image.open(path).verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


@torch.no_grad()
def evaluate_retrieval_top1(
    model,
    samples,
    tokenizer,
    device,
    data_path: str,
    max_negatives: int = 31,
    max_samples: int | None = None,
    seed: int = 42,
) -> float:
    random.seed(seed)
    torch.manual_seed(seed)

    transform = build_image_transform(image_size=224)

    valid_samples = []
    for sample in samples:
        image_path = resolve_image_path(sample["image_path"], data_path)
        if is_valid_image(image_path):
            valid_samples.append(sample)

    random.shuffle(valid_samples)
    if max_samples is not None:
        valid_samples = valid_samples[:max_samples]

    if len(valid_samples) == 0:
        return 0.0

    all_texts = [sample["text"] for sample in valid_samples]

    correct = 0
    total = 0

    for sample in tqdm(valid_samples, desc="eval"):
        image_path = resolve_image_path(sample["image_path"], data_path)
        if not is_valid_image(image_path):
            continue

        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        negatives = [t for t in all_texts if t != sample["text"]]
        random.shuffle(negatives)
        negatives = negatives[:max_negatives]

        candidates = [sample["text"]] + negatives
        tokens = torch.tensor(
            tokenizer.batch_pad(tokenizer.batch_encode(candidates)),
            dtype=torch.long,
            device=device,
        )

        logits_per_image, _ = model(image, tokens)
        pred = logits_per_image.argmax(dim=-1).item()

        correct += int(pred == 0)
        total += 1

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/val.json")
    parser.add_argument("--ckpt", type=str, default="checkpoints/miniclip.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--num-negatives", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.tokenizer, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tokenizer = SimpleTokenizer(vocab=vocab, config=TokenizerConfig(max_length=32))
    model = MiniCLIP(vocab_size=tokenizer.vocab_size, max_length=32, embed_dim=256).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    with open(args.data, "r", encoding="utf-8") as f:
        samples = json.load(f)

    acc = evaluate_retrieval_top1(
        model=model,
        samples=samples,
        tokenizer=tokenizer,
        device=device,
        data_path=args.data,
        max_negatives=args.num_negatives,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"retrieval_top1_acc={acc:.4f}")


if __name__ == "__main__":
    main()
