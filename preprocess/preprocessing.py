from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32

CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes',
    'trash', 'white-glass',
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

# những ảnh bị gán sai nhãn, đã xem bằng mắt ở bước EDA
LABEL_CORRECTIONS = {
    'metal/metal459.jpg': 'white-glass',
    'plastic/plastic159.jpg': 'white-glass',
    'white-glass/white-glass761.jpg': 'plastic',
}

# xác suất và mức độ của từng phép augmentation
HFLIP_P = 0.5
VFLIP_P = 0.2
ROTATION_P = 0.5
ROTATION_DEGREES = 15
BRIGHTNESS_P = 0.5
BRIGHTNESS_DELTA = 0.15
CONTRAST_P = 0.5
CONTRAST_DELTA = 0.15

project_dir = Path(__file__).resolve().parent.parent
DATA_DIR = project_dir / 'data' / 'raw' / 'garbage_classification'
CSV_DIR = project_dir / 'data'


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# đọc danh sách ảnh, bỏ qua ảnh lỗi
def scan_images(data_dir):
    records = []
    for path in sorted(data_dir.glob('*/*')):
        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            with Image.open(path) as image:
                image.load()
        except Exception:
            print('Bỏ qua ảnh lỗi:', path)
            continue
        records.append({
            'filepath': path.relative_to(data_dir).as_posix(),
            'label': path.parent.name,
            'file_hash': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    data = pd.DataFrame(records)
    print('Số ảnh đọc được:', len(data))
    return data


def fix_labels(data):
    for filepath, label in LABEL_CORRECTIONS.items():
        data.loc[data['filepath'] == filepath, 'label'] = label
    return data


# bỏ ảnh trùng, mỗi nhóm trùng chỉ giữ lại 1 ảnh
def remove_duplicates(data):
    label_count = data.groupby('file_hash')['label'].transform('nunique')
    if (label_count > 1).any():
        print(data[label_count > 1][['filepath', 'label']])
        raise ValueError('Còn ảnh trùng nhưng khác nhãn, cần sửa LABEL_CORRECTIONS')

    before = len(data)
    data = data.sort_values('filepath').drop_duplicates(subset='file_hash')
    data = data.reset_index(drop=True)
    print('Số ảnh trùng đã bỏ:', before - len(data))
    return data


# chia 70% train, 15% val, 15% test theo tỉ lệ từng lớp
def split_data(data):
    train_data, temp_data = train_test_split(
        data, test_size=0.3, stratify=data['label'], random_state=SEED)
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=temp_data['label'], random_state=SEED)
    print('Train:', len(train_data), '| Val:', len(val_data), '| Test:', len(test_data))
    return train_data, val_data, test_data


def save_splits(train_data, val_data, test_data):
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        folder = CSV_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        data.to_csv(folder / (name + '.csv'), index=False)


def run_preprocessing():
    data = scan_images(DATA_DIR)
    data = fix_labels(data)
    data = remove_duplicates(data)
    print(data['label'].value_counts())

    train_data, val_data, test_data = split_data(data)
    save_splits(train_data, val_data, test_data)


# trọng số lớp tính từ tập train, dùng cho CrossEntropyLoss(weight=...)
def compute_class_weights():
    labels = pd.read_csv(CSV_DIR / 'train' / 'train.csv')['label']
    counts = labels.value_counts().reindex(CLASSES)
    weights = len(labels) / (len(CLASSES) * counts.to_numpy())
    return torch.tensor(weights, dtype=torch.float32)


# resize giữ tỉ lệ, phần thiếu thêm viền đen cho thành ảnh vuông
class ResizeWithPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize((new_width, new_height), Image.BILINEAR)

        result = Image.new('RGB', (self.size, self.size), (0, 0, 0))
        result.paste(image, ((self.size - new_width) // 2, (self.size - new_height) // 2))
        return result


# augmentation chỉ dùng cho tập train, mỗi phép có xác suất riêng
def build_augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=HFLIP_P),
        transforms.RandomVerticalFlip(p=VFLIP_P),
        transforms.RandomApply([
            transforms.RandomRotation(ROTATION_DEGREES, interpolation=transforms.InterpolationMode.BILINEAR)
        ], p=ROTATION_P),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=BRIGHTNESS_DELTA)
        ], p=BRIGHTNESS_P),
        transforms.RandomApply([
            transforms.ColorJitter(contrast=CONTRAST_DELTA)
        ], p=CONTRAST_P),
    ])


class GarbageDataset(Dataset):
    def __init__(self, csv_path, img_size=IMG_SIZE, training=False):
        data = pd.read_csv(csv_path)
        self.filepaths = data['filepath'].tolist()
        self.labels = data['label'].map(CLASS_TO_INDEX).tolist()
        self.resize = ResizeWithPad(img_size)
        self.to_tensor = transforms.ToTensor()
        self.augmentation = build_augmentation() if training else None

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        # 1. đọc ảnh
        path = DATA_DIR / self.filepaths[index]
        image = Image.open(path).convert('RGB')
        # 2. resize
        image = self.resize(image)
        # 3. augmentation (chỉ tập train)
        if self.augmentation is not None:
            image = self.augmentation(image)
        # 4. chuyển thành tensor, pixel về [0, 1]
        image = self.to_tensor(image)
        return image, self.labels[index]


def make_dataloaders(img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=0):
    set_seed()
    train_data = GarbageDataset(CSV_DIR / 'train' / 'train.csv', img_size, training=True)
    val_data = GarbageDataset(CSV_DIR / 'val' / 'val.csv', img_size)
    test_data = GarbageDataset(CSV_DIR / 'test' / 'test.csv', img_size)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    run_preprocessing()

    train_loader, val_loader, test_loader = make_dataloaders()
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        images, labels = next(iter(loader))
        print(name, '| Shape:', tuple(images.shape), '| Pixel:', images.min().item(), '-', images.max().item())

    print('Trọng số lớp:', compute_class_weights())