# Mini-CLIP

A minimal CLIP reproduction in PyTorch.

This repository focuses on the core CLIP pipeline:

- text tokenizer
- image encoder
- text encoder
- contrastive loss
- training and retrieval evaluation

It is designed to be easy to read, easy to run, and easy to extend.

## Tested Environment

- GPU: NVIDIA RTX 5090 32GB x1
- Recommended Python: 3.10+
- Recommended PyTorch stack: CUDA 12.8 wheels (`cu128`)

## Installation

Create a clean environment first:

```bash
conda create -n mini_clip python=3.10 -y
conda activate mini_clip
pip install -r requirements.txt
```

If you are on an RTX 5090, use the CUDA 12.8 build of PyTorch pinned in `requirements.txt`.

## Data Format

The training code expects JSON files with image-text pairs:

```json
[
  {
    "image_path": "raw/COCO2017/train2017/000000000009.jpg",
    "text": "a man riding a bicycle"
  }
]
```

Typical split files:

- `data/train.json`
- `data/val.json`
- `data/test.json`

## Training

```bash
python train.py \
  --data data/train.json \
  --val-data data/val.json \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 4
```

Recommended starting values for a single RTX 5090 32GB:

- `batch-size=32`
- `num-workers=4`
- `epochs=10`
- `lr=1e-4`

Training saves the best checkpoint to `checkpoints/miniclip.pt` and the tokenizer vocabulary to `checkpoints/tokenizer.json`.

## Evaluation

```bash
python eval.py \
  --data data/val.json \
  --ckpt checkpoints/miniclip.pt \
  --tokenizer checkpoints/tokenizer.json \
  --max-samples 50 \
  --num-negatives 31
```

Evaluation results can be written to `results/` for later comparison.

## Demo

```bash
python demo.py \
  --image CLIP.png \
  --texts "a dog" "a cat" "a diagram" \
  --ckpt checkpoints/miniclip.pt \
  --tokenizer checkpoints/tokenizer.json
```

## Notes

- This is an educational implementation, not the original OpenAI CLIP codebase.
- Use a real image-text dataset such as COCO Captions for better results.
- `checkpoints/`, `results/`, and raw datasets should stay out of git.

## Repository Layout

```text
README.md
requirements.txt
train.py
demo.py
src/
  tokenizer.py
  preprocess.py
  image_encoder.py
  text_encoder.py
  model.py
  loss.py
  data.py
  utils.py
```
