
"""Đọc CSV train/val/test và tạo DataLoader cho MobileNetV3."""

import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 1. Settings
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

IMAGE_SIZE = 224
BATCH_SIZE = 32

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# 2. Dataset
class WasteDataset(Dataset):
    def __init__(self, rows, image_dir, class_to_index, training=False):
        self.rows = rows 
        # {"filepath": "train/0/0001.jpg", "label": "cardboard"} for row in rows
        self.image_dir = image_dir
        self.class_to_index = class_to_index

        if training:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomAffine(
                    degrees=20,
                    scale=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.15
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.image_dir / row["filepath"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)
        label = self.class_to_index[row["label"]]

        return image, label


# 3. Read CSV
def read_csv(file_path):
    with open(file_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# 4. Calculate class weights
def compute_class_weights(train_dataset, classes):
    labels = [row["label"] for row in train_dataset.rows]
    counts = Counter(labels)

    total_samples = len(labels)
    num_classes = len(classes)

    weights = []

    for class_name in classes:
        count = counts[class_name]
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


# 5. Prepare DataLoaders
def prepare(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    data_dir = Path(data_dir)
    image_dir = data_dir / "raw" / "garbage_classification"

    train_rows = read_csv(data_dir / "train" / "train.csv")
    val_rows = read_csv(data_dir / "val" / "val.csv")
    test_rows = read_csv(data_dir / "test" / "test.csv")

    # Tạo class index từ training set
    classes = sorted({row["label"] for row in train_rows})
    class_to_index = {name: i for i, name in enumerate(classes)}

    train_dataset = WasteDataset(
        train_rows, image_dir, class_to_index, training=True
    )

    val_dataset = WasteDataset(
        val_rows, image_dir, class_to_index
    )

    test_dataset = WasteDataset(
        test_rows, image_dir, class_to_index
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    return loaders, classes


# 6. Check
if __name__ == "__main__":
    loaders, classes = prepare()

    class_weights = compute_class_weights(
        loaders["train"].dataset, classes
    )

    print("Classes:", classes)
    print("Train:", len(loaders["train"].dataset))
    print("Validation:", len(loaders["val"].dataset))
    print("Test:", len(loaders["test"].dataset))
    print("Class weights:", class_weights)
