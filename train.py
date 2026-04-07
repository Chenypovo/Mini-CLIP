from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.optim import AdamW
from tqdm import tqdm

from src.data import build_dataloader, build_tokenizer_from_json
from src.loss import clip_contrasive_loss
from src.model import MiniCLIP
from src.tokenizer import TokenizerConfig


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
    dataloader,
    tokenizer,
    device,
    data_path: str,
    max_negatives: int = 31,
    max_samples: int | None = None,
    seed: int = 42,
) -> float:
    random.seed(seed)
    torch.manual_seed(seed)

    transform = dataloader.dataset.transform
    samples = list(dataloader.dataset.samples)

    valid_samples = []
    for sample in samples:
        image_path = resolve_image_path(sample.image_path, data_path)
        if is_valid_image(image_path):
            valid_samples.append(sample)

    random.shuffle(valid_samples)
    if max_samples is not None:
        valid_samples = valid_samples[:max_samples]

    if len(valid_samples) == 0:
        return 0.0

    all_texts = [sample.text for sample in valid_samples]

    correct = 0
    total = 0

    for sample in tqdm(valid_samples, desc="val", leave=False):
        image_path = resolve_image_path(sample.image_path, data_path)
        if not is_valid_image(image_path):
            continue

        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        negatives = [t for t in all_texts if t != sample.text]
        random.shuffle(negatives)
        negatives = negatives[:max_negatives]

        candidates = [sample.text] + negatives
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
    parser.add_argument("--data", type=str, default="data/train.json")
    parser.add_argument("--val-data", type=str, default="data/val.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="checkpoints/miniclip.pt")
    parser.add_argument("--tokenizer-path", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--max-val-samples", type=int, default=50)
    parser.add_argument("--max-negatives", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(args.data)
    val_path = Path(args.val_data)

    tokenizer = build_tokenizer_from_json(
        str(data_path),
        config=TokenizerConfig(max_length=args.max_length),
    )

    train_loader = build_dataloader(
        str(data_path),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    val_loader = build_dataloader(
        str(val_path),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    model = MiniCLIP(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        embed_dim=args.embed_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_val = -1.0

    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            images = batch["images"].to(device, non_blocking=True)
            texts = batch["texts"].to(device, non_blocking=True)

            logits_per_image, logits_per_text = model(images, texts)
            loss = clip_contrasive_loss(logits_per_image, logits_per_text)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_acc = evaluate_retrieval_top1(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            data_path=str(val_path),
            max_negatives=args.max_negatives,
            max_samples=args.max_val_samples,
            seed=args.seed,
        )
        print(f"epoch {epoch + 1}/{args.epochs} val_top1_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            with open(args.tokenizer_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=2)
            torch.save(model.state_dict(), args.save_path)
            print(f"saved best model to {args.save_path}")
            print(f"saved tokenizer to {args.tokenizer_path}")


if __name__ == "__main__":
    main()
