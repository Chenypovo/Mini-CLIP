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

## Data Preparation

### Option 1: COCO2017 (recommended)

1. Download [COCO 2017](https://cocodataset.org/#download) images and captions:

```bash
# Example: download to data/raw/COCO2017/
mkdir -p data/raw/COCO2017
cd data/raw/COCO2017

# Train
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip

# Val
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

2. Prepare JSON splits:

```bash
python prepare_coco2017.py
```

This generates `data/train.json` and `data/val.json`.

### Option 2: Flickr8k

1. Download [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) to `data/raw/flickr8k/`:

```text
data/raw/flickr8k/
  Images/          # image files
  captions.txt     # captions file
```

2. Prepare JSON splits:

```bash
python prepare_flickr8k.py
```

This generates `data/train.json`, `data/val.json`, and `data/test.json`.

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

Retrieval evaluation is built into `train.py` and runs automatically after each epoch. You can also call `evaluate_retrieval_top1()` directly from `train.py`.

Results can be written to `results/` for later comparison.

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
- `checkpoints/`, `results/`, and raw datasets should stay out of git (see `.gitignore`).

## Repository Layout

```text
Mini-CLIP/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py                  # training loop + inline evaluation
├── demo.py                   # inference demo
├── prepare_coco2017.py       # COCO2017 dataset preparation
├── prepare_flickr8k.py       # Flickr8k dataset preparation
├── src/
│   ├── tokenizer.py          # BPE tokenizer
│   ├── preprocess.py         # data preprocessing
│   ├── image_encoder.py      # ViT-based image encoder
│   ├── text_encoder.py       # text encoder
│   ├── model.py              # MiniCLIP model
│   ├── loss.py               # contrastive loss
│   └── data.py               # dataset & dataloader
├── data/                     # (gitignored)
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── raw/
│       ├── COCO2017/
│       │   ├── train2017/
│       │   ├── val2017/
│       │   └── annotations/
│       └── flickr8k/
│           ├── Images/
│           └── captions.txt
├── checkpoints/              # (gitignored)
└── results/                  # (gitignored)
```
