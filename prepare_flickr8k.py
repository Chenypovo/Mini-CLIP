from pathlib import Path
import json
import random
from PIL import Image
root = Path("data/raw/flickr8k")
img_dir = root / "Images"
captions_path = root / "captions.txt"

random.seed(42)

def read_captions(path):
    captions = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                left, cap = line.split("\t", 1)
            elif "," in line:
                left, cap = line.split(",", 1)
            else:
                continue

            img_name = left.split("#")[0].strip()
            cap = cap.strip()
            captions.setdefault(img_name, []).append(cap)

    return captions

captions = read_captions(captions_path)
image_names = sorted(captions.keys())
random.shuffle(image_names)

n = len(image_names)
n_train = int(n * 0.7)
n_val = int(n * 0.2)

train_imgs = image_names[:n_train]
val_imgs = image_names[n_train:n_train + n_val]
test_imgs = image_names[n_train + n_val:]

def is_valid_image(path):
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False

def build(samples, out_path):
    items = []
    for img_name in samples:
        img_path = img_dir / img_name
        if not is_valid_image(img_path):
            print("skip bad image:", img_path)
            continue
        for cap in captions.get(img_name, []):
            items.append({
                "image_path": str(Path("raw/flickr8k/Images") / img_name),
                "text": cap,
            })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

build(train_imgs, "data/train.json")
build(val_imgs, "data/val.json")
build(test_imgs, "data/test.json")

print("done")
print("train:", len(train_imgs), "val:", len(val_imgs), "test:", len(test_imgs))
