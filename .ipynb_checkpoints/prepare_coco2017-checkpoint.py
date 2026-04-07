from __future__ import annotations

import json
import random
from pathlib import Path
from PIL import Image, UnidentifiedImageError

random.seed(42)

root = Path("/root/autodl-tmp/mini-clip/data/raw/COCO2017")
train_img_dir = root / "train2017"
val_img_dir = root / "val2017"
train_ann_path = root / "annotations" / "captions_train2017.json"
val_ann_path = root / "annotations" / "captions_val2017.json"


def is_valid_image(path: Path) -> bool:
    try:
        Image.open(path).verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def load_coco_captions(ann_path: Path):
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    captions = {}
    for ann in data["annotations"]:
        file_name = id_to_file[ann["image_id"]]
        captions.setdefault(file_name, []).append(ann["caption"])

    return captions


def build_split(img_dir: Path, ann_path: Path, out_path: Path, max_caps_per_image: int = 1):
    captions = load_coco_captions(ann_path)
    items = []

    for file_name, caps in captions.items():
        img_path = img_dir / file_name
        if not is_valid_image(img_path):
            print("skip bad image:", img_path)
            continue

        random.shuffle(caps)
        caps = caps[:max_caps_per_image]

        for cap in caps:
            items.append({
                "image_path": str(Path("raw/COCO2017") / img_dir.name / file_name),
                "text": cap,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"{out_path}: {len(items)} samples")


def main():
    build_split(train_img_dir, train_ann_path, Path("data/train.json"), max_caps_per_image=1)
    build_split(val_img_dir, val_ann_path, Path("data/val.json"), max_caps_per_image=1)


if __name__ == "__main__":
    main()
