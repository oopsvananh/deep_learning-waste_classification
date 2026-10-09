from pathlib import Path
import hashlib

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 224
BATCH_SIZE = 32

CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes',
    'trash', 'white-glass',
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

# images with wrong labels, checked by eye in the EDA step
LABEL_CORRECTIONS = {
    'metal/metal459.jpg': 'white-glass',
    'plastic/plastic159.jpg': 'white-glass',
    'white-glass/white-glass761.jpg': 'plastic',
}

# probability and strength of each augmentation
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


# check if an image can be opened
def is_valid_image(path):
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False
    
# read the image list, skip broken images
def scan_images(data_dir):
    records = []
    for path in sorted(data_dir.glob('*/*')):
        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        if not is_valid_image(path):
            print('Skipping broken image:', path)
            continue
        records.append({
            'filepath': path.relative_to(data_dir).as_posix(),
            'label': path.parent.name,
            'file_hash': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    data = pd.DataFrame(records)
    print('Images read:', len(data))
    return data


def fix_labels(data):
    for filepath, label in LABEL_CORRECTIONS.items():
        data.loc[data['filepath'] == filepath, 'label'] = label
    return data


# remove duplicate images, keep only one image per duplicate group
def remove_duplicates(data):
    # 1. same image but different labels: stop, fix the labels first
    n_labels = data.groupby('file_hash')['label'].transform('nunique')
    conflict = data[n_labels > 1]
    if len(conflict) > 0:
        print(conflict[['filepath', 'label']])
        raise ValueError('Duplicate images with different labels remain, fix LABEL_CORRECTIONS')
    
    before = len(data)
    data = data.sort_values('filepath').drop_duplicates(subset='file_hash')
    data = data.reset_index(drop=True)
    print('Duplicates removed:', before - len(data))
    return data


# split 70% train, 15% val, 15% test, keeping the class proportions
def split_data(data):
    train_data, temp_data = train_test_split(
        data, test_size=0.3, stratify=data['label'])
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=temp_data['label'])
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


# class weights computed from the train set, used for CrossEntropyLoss(weight=...)
def compute_class_weights(train_csv_path=None):
    train_csv_path = Path(train_csv_path or CSV_DIR / 'train' / 'train.csv')
    labels = pd.read_csv(train_csv_path)['label']
    counts = labels.value_counts().reindex(CLASSES)
    weights = len(labels) / (len(CLASSES) * counts.to_numpy())
    return torch.tensor(weights, dtype=torch.float32)


# resize keeping the aspect ratio, pad with black borders to make a square image
def resize_with_pad(image, size):
    width, height = image.size
    scale = size / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    image = image.resize((new_width, new_height), Image.BILINEAR)

    result = Image.new('RGB', (size, size), (0, 0, 0))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    result.paste(image, (left, top))
    return result

# augmentation is only used for the train set, each one has its own probability
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
    def __init__(self, csv_path, img_size=IMG_SIZE, training=False, data_dir=None):
        data = pd.read_csv(csv_path)
        self.filepaths = data['filepath'].tolist()
        self.labels = data['label'].map(CLASS_TO_INDEX).tolist()
        self.data_dir = Path(data_dir or DATA_DIR)
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()
        self.augmentation = build_augmentation() if training else None

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        # 1. read the image
        path = self.data_dir / self.filepaths[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
        # 2. resize
        image = resize_with_pad(image, self.img_size)
        # 3. augmentation (train set only)
        if self.augmentation is not None:
            image = self.augmentation(image)
        # 4. convert to tensor, pixels scaled to [0, 1]
        image = self.to_tensor(image)
        return image, self.labels[index]


def make_dataloaders(img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=2,
                     data_dir=None, csv_dir=None):
    """Tạo ba DataLoader; chỉ tập train được shuffle và augmentation."""
    data_dir = Path(data_dir or DATA_DIR)
    csv_dir = Path(csv_dir or CSV_DIR)

    train_data = GarbageDataset(
        csv_dir / 'train' / 'train.csv', img_size, training=True, data_dir=data_dir)
    val_data = GarbageDataset(
        csv_dir / 'val' / 'val.csv', img_size, data_dir=data_dir)
    test_data = GarbageDataset(
        csv_dir / 'test' / 'test.csv', img_size, data_dir=data_dir)

    loader_options = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    run_preprocessing()

    train_loader, val_loader, test_loader = make_dataloaders()
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        images, labels = next(iter(loader))
        print(name, '| Shape:', tuple(images.shape), '| Pixel:', images.min().item(), '-', images.max().item())

    print('Class weights:', compute_class_weights())
