
import os
import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 1. Parameters
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32

CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]

CLASS_TO_INDEX = {}
for i, name in enumerate(CLASSES):
    CLASS_TO_INDEX[name] = i

LABEL_CORRECTIONS = {
    'metal/metal459.jpg': 'white-glass',
    'plastic/plastic159.jpg': 'white-glass',
    'white-glass/white-glass761.jpg': 'plastic'
}

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

torch.manual_seed(SEED)


# 2. Read image information
def scan_images(data_dir):
    records = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            try:
                with Image.open(img_path) as img:
                    img.load()
            except Exception:
                print("Cannot open:", img_path)
                continue
            
            filepath = folder.name + '/' + img_path.name # image information
            label = folder.name
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest() # create hash to find duplicate images
            records.append({
                'filepath': filepath,
                'label': label,
                'file_hash': file_hash
            })
    data = pd.DataFrame(records)
    print("Total images:", len(data))
    return data


# 3. Correct wrong labels
def fix_labels(data):
    for filepath, new_label in LABEL_CORRECTIONS.items():
        data.loc[data['filepath'] == filepath, 'label'] = new_label
    return data


# 4. Remove duplicate images
def remove_duplicates(data):
    # Check duplicate images with different labels
    for file_hash, group in data.groupby('file_hash'):
        labels = group['label'].unique()
        if len(labels) > 1:
            print(group[['filepath', 'label']])
            raise ValueError("Duplicate images have different labels")
    # Remove duplicates
    before = len(data)
    data = data.drop_duplicates(subset='file_hash', keep='first')
    data = data.reset_index(drop=True)
    print("Duplicates removed:", before - len(data))
    return data


# 5. Split train, validation and test sets
def split_data(data):
    train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=SEED)
    print("Train set:", len(train_data))
    print("Validation set:", len(val_data))
    print("Test set:", len(test_data))
    return train_data, val_data, test_data


# 6. Save datasets to CSV
def save_splits(train_data, val_data, test_data):
    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    for name, data in datasets.items():
        folder = CSV_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / (name + '.csv')
        data.to_csv(csv_path, index=False)


# 7. Run preprocessing
def run_preprocessing():
    data = scan_images(DATA_DIR)
    data = fix_labels(data)
    data = remove_duplicates(data)
    print(data['label'].value_counts())
    train_data, val_data, test_data = split_data(data)
    save_splits(train_data, val_data, test_data)


# 8. Calculate class weights
def compute_class_weights():
    train_data = pd.read_csv(CSV_DIR / 'train' / 'train.csv')
    counts = train_data['label'].value_counts()
    weights = []

    for class_name in CLASSES:
        count = counts[class_name]
        weight = len(train_data) / (len(CLASSES) * count)
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


# 9. Resize image and add padding
def resize_with_pad(image, size):
    width, height = image.size
    # New image size
    scale = size / max(width, height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    image = image.resize((new_width, new_height), Image.BILINEAR)
    # Create a black square background
    new_image = Image.new('RGB', (size, size), (0, 0, 0))
    # Put the image in the center
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    new_image.paste(image, (left, top))
    return new_image


# 10. Data augmentation
def build_augmentation():
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=HFLIP_P),
        transforms.RandomVerticalFlip(p=VFLIP_P),
        transforms.RandomApply([
            transforms.RandomRotation(
                ROTATION_DEGREES,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        ], p=ROTATION_P),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=BRIGHTNESS_DELTA)
        ], p=BRIGHTNESS_P),
        transforms.RandomApply([
            transforms.ColorJitter(contrast=CONTRAST_DELTA)
        ], p=CONTRAST_P)
    ])
    return augmentation


# 11. Create Dataset
class GarbageDataset(Dataset):
    def __init__(self, csv_path, img_size=IMG_SIZE, training=False):
        data = pd.read_csv(csv_path)
        self.filepaths = data['filepath'].tolist()
        self.labels = []
        
        for label in data['label']:
            self.labels.append(CLASS_TO_INDEX[label])
        self.img_size = img_size
        self.training = training
        self.to_tensor = transforms.ToTensor()
        self.augmentation = build_augmentation()
    def __len__(self):
        return len(self.filepaths)
    def __getitem__(self, index):
        # Load image
        img_path = DATA_DIR / self.filepaths[index]
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        # Resize image
        image = resize_with_pad(image, self.img_size)
        # Apply augmentation to training data only
        if self.training:
            image = self.augmentation(image)
        # Convert image to tensor
        image = self.to_tensor(image)
        label = self.labels[index]
        return image, label


# 12. Create DataLoaders
def make_dataloaders():
    train_data = GarbageDataset(CSV_DIR / 'train' / 'train.csv', training=True)
    val_data = GarbageDataset(CSV_DIR / 'val' / 'val.csv')
    test_data = GarbageDataset(CSV_DIR / 'test' / 'test.csv')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


# 13. Run
if __name__ == '__main__':
    run_preprocessing()
    train_loader, val_loader, test_loader = make_dataloaders()
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        images, labels = next(iter(loader))
        print(name)
        print("Image shape:", images.shape)
        print("Min pixel:", images.min().item())
        print("Max pixel:", images.max().item())
    print("Class weights:", compute_class_weights())
