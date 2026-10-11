import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
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
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

LABEL_CORRECTIONS = {
    'metal/metal459.jpg': 'white-glass',
    'plastic/plastic159.jpg': 'white-glass',
    'white-glass/white-glass761.jpg': 'plastic'
}

project_dir = Path(__file__).resolve().parent.parent
DATA_DIR = project_dir / 'data' / 'raw' / 'garbage_classification'
CSV_DIR = project_dir / 'data'

torch.manual_seed(SEED)


# 2. Read image information (filepath, label, hash)
def scan_images():
    records = []
    for img_path in sorted(DATA_DIR.glob('*/*')):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            with Image.open(img_path) as img:
                img.load()  # check image is not corrupted
        except Exception:
            print("Cannot open:", img_path)
            continue
        records.append({
            'filepath': f'{img_path.parent.name}/{img_path.name}',
            'label': img_path.parent.name,
            'file_hash': hashlib.md5(img_path.read_bytes()).hexdigest()
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

# 6. Class weights
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


# 7. Augmentation
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
])

class GarbageDataset(Dataset):
    def __init__(self, csv_path, training=False):
        data = pd.read_csv(csv_path)
        self.filepaths = data['filepath'].tolist()
        self.labels = [CLASS_TO_INDEX[label] for label in data['label']]
        self.training = training
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        with Image.open(DATA_DIR / self.filepaths[index]) as img:
            image = img.convert('RGB')
        # Resize keeping aspect ratio, pad with black to a square, method=Image.BILINEAR: cho ảnh mượt vừa phải, nhanh và chất lượng ổn
        image = ImageOps.pad(image, (IMG_SIZE, IMG_SIZE), method=Image.BILINEAR, color=(0, 0, 0)) # 0 0 0 là màu nền lấp chỗ trống, màu đen
        if self.training:
            image = augmentation(image)
        return self.to_tensor(image), self.labels[index]


# 7. DataLoaders
def make_dataloaders():
    train = GarbageDataset(CSV_DIR / 'train' / 'train.csv', training=True)
    val = GarbageDataset(CSV_DIR / 'val' / 'val.csv')
    test = GarbageDataset(CSV_DIR / 'test' / 'test.csv')
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val, batch_size=BATCH_SIZE),
        DataLoader(test, batch_size=BATCH_SIZE),
    )


# 8. Run
if __name__ == '__main__':
    run_preprocessing()
    loaders = make_dataloaders()
    for name, loader in zip(['train', 'val', 'test'], loaders):
        images, labels = next(iter(loader))
        print(name, "| shape:", images.shape, "| min:", images.min().item(), "| max:", images.max().item())
    print("Class weights:", compute_class_weights())