from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.model import MiniCLIP
from src.preprocess import build_image_transform
from src.tokenizer import SimpleTokenizer, TokenizerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--texts", type=str, nargs="+", required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/miniclip.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.tokenizer, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tokenizer = SimpleTokenizer(vocab=vocab, config=TokenizerConfig(max_length=32))

    model = MiniCLIP(
        vocab_size=tokenizer.vocab_size,
        max_length=32,
        embed_dim=256,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    transform = build_image_transform(image_size=224)
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    tokens = torch.tensor(
        tokenizer.batch_pad(tokenizer.batch_encode(args.texts)),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits_per_image, _ = model(image, tokens)
        probs = logits_per_image.softmax(dim=-1)

    print("texts:", args.texts)
    print("probabilities:", probs.cpu().numpy())


if __name__ == "__main__":
    main()
