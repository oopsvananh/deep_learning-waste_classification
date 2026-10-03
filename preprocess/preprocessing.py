"""Tiền xử lý ảnh rác và tạo DataLoader cho CNN 12 lớp.

Chạy file để kiểm tra ảnh, sửa nhãn, loại trùng và lưu CSV train/val/test.
Ảnh được resize giữ tỉ lệ, thêm viền và chuyển pixel về [0, 1].
Augmentation chỉ áp dụng cho tập train.
"""

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


SEED = 42
IMG_SIZE = 224      # kích thước ảnh đầu vào cho Simple/Complex CNN
BATCH_SIZE = 32
VAL_RATIO = 0.15
TEST_RATIO = 0.15
EXTENSIONS = {'.jpg', '.jpeg', '.png'}

CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes',
    'trash', 'white-glass',
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}

LABEL_CORRECTIONS = {
    'metal/metal459.jpg': 'white-glass',
    'plastic/plastic159.jpg': 'white-glass',
    'white-glass/white-glass761.jpg': 'plastic',
}


def find_data_dir():
    """Tìm thư mục ảnh từ vị trí file hoặc thư mục làm việc."""
    here = Path(__file__).resolve().parent
    bases = [here, here.parent, Path.cwd(), *Path.cwd().parents]
    relatives = [
        'data/raw/garbage_classification',
        'deep_learning-waste_classification/data/raw/garbage_classification',
    ]
    for base in bases:
        for relative in relatives:
            candidate = base / relative
            if candidate.is_dir():
                return candidate.resolve()
    raise FileNotFoundError(
        'Không tìm thấy thư mục data/raw/garbage_classification. '
        'Hãy kiểm tra lại cấu trúc thư mục hoặc truyền data_dir thủ công.'
    )


def get_csv_dir(data_dir=None):
    return Path(data_dir or find_data_dir()).parent.parent


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_images(data_dir):
    """Kiểm tra ảnh đọc được và tính hash để tìm file trùng từng byte."""
    data_dir = Path(data_dir)
    records = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    im.load()
                records.append({
                    'filepath': img_path.relative_to(data_dir).as_posix(),
                    'label': class_dir.name,
                    'file_hash': file_sha256(img_path),
                })
            except Exception as exc:
                print(f'Bỏ qua ảnh lỗi: {img_path} ({exc})')

    if not records:
        raise RuntimeError('Không có ảnh hợp lệ.')
    df = pd.DataFrame(records)
    unknown = set(df['label']) - set(CLASSES)
    if unknown:
        raise ValueError(f'Thư mục lớp không nằm trong CLASSES: {sorted(unknown)}')
    print(f'[scan] Tổng số ảnh hợp lệ: {len(df):,}')
    return df


def find_cross_label_duplicates(df):
    groups = []
    for file_hash, members in df.groupby('file_hash'):
        if len(members) > 1 and members['label'].nunique() > 1:
            groups.append(list(zip(members['filepath'], members['label'])))
    return groups


def apply_label_corrections(df, corrections=None):
    corrections = LABEL_CORRECTIONS if corrections is None else corrections
    df = df.copy()
    log = []
    for filepath, correct_label in corrections.items():
        if correct_label not in CLASS_TO_INDEX:
            raise ValueError(f'Nhãn "{correct_label}" của {filepath} không nằm trong CLASSES.')
        mask = df['filepath'] == filepath
        if not mask.any():
            print(f'[labels] Cảnh báo: không tìm thấy {filepath}, bỏ qua.')
            continue
        old_label = df.loc[mask, 'label'].iloc[0]
        df.loc[mask, 'label'] = correct_label
        log.append({'filepath': filepath, 'old_label': old_label, 'new_label': correct_label})
    print(f'[labels] Đã sửa nhãn {len(log)} ảnh.')
    return df, pd.DataFrame(log, columns=['filepath', 'old_label', 'new_label'])


def remove_same_label_duplicates(df):
    """Giữ một file trong mỗi nhóm trùng cùng nhãn; dừng nếu nhãn mâu thuẫn."""
    drop_indices, log = [], []
    for file_hash, members in df.groupby('file_hash'):
        if len(members) < 2:
            continue
        members = members.sort_values('filepath')
        if members['label'].nunique() > 1:
            print(f'[dedup] Cảnh báo: nhóm {file_hash[:8]}... vẫn khác nhãn, chưa xử lý: '
                  f'{members["filepath"].tolist()}')
            continue
        keep, *drop = members.index.tolist()
        for idx in drop:
            drop_indices.append(idx)
            log.append({
                'removed_filepath': df.loc[idx, 'filepath'],
                'kept_filepath': df.loc[keep, 'filepath'],
                'label': df.loc[idx, 'label'],
            })

    cleaned = df.drop(index=drop_indices).reset_index(drop=True)
    if not cleaned['file_hash'].is_unique:
        raise RuntimeError('Còn ảnh trùng chưa xử lý (nhóm khác nhãn). Kiểm tra lại LABEL_CORRECTIONS.')
    print(f'[dedup] Đã bỏ {len(drop_indices)} ảnh trùng cùng nhãn, còn {len(cleaned):,} ảnh.')
    return cleaned, pd.DataFrame(log, columns=['removed_filepath', 'kept_filepath', 'label'])


def split_dataset(df, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED):
    """Chia tập theo tỉ lệ lớp và kiểm tra trùng giữa các tập."""
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError('val_ratio và test_ratio phải trong (0, 1) và tổng < 1.')

    train_df, temp_df = train_test_split(
        df, test_size=val_ratio + test_ratio, stratify=df['label'], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio / (val_ratio + test_ratio), stratify=temp_df['label'], random_state=seed)

    parts = [train_df, val_df, test_df]
    if sum(len(p) for p in parts) != len(df):
        raise RuntimeError('Mất bản ghi khi chia tập.')
    for column in ('filepath', 'file_hash'):
        for i in range(3):
            for j in range(i + 1, 3):
                if not set(parts[i][column]).isdisjoint(parts[j][column]):
                    raise ValueError(f'Trùng {column} giữa các tập.')

    print(f'[split] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, csv_dir, label_log=None, dedup_log=None):
    csv_dir = Path(csv_dir)
    for name, part in [('train', train_df), ('val', val_df), ('test', test_df)]:
        (csv_dir / name).mkdir(parents=True, exist_ok=True)
        part.to_csv(csv_dir / name / f'{name}.csv', index=False, encoding='utf-8-sig')

    processed_dir = csv_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    if label_log is not None:
        label_log.to_csv(processed_dir / 'label_corrections_applied.csv', index=False, encoding='utf-8-sig')
    if dedup_log is not None:
        dedup_log.to_csv(processed_dir / 'removed_duplicates.csv', index=False, encoding='utf-8-sig')
    print(f'[save] Đã lưu CSV vào {csv_dir}')


def run_preprocessing(data_dir=None, seed=SEED):
    data_dir = Path(data_dir or find_data_dir())
    csv_dir = get_csv_dir(data_dir)

    df = scan_images(data_dir)

    cross_groups = find_cross_label_duplicates(df)
    if cross_groups:
        print(f'[labels] {len(cross_groups)} nhóm trùng khác nhãn (đối chiếu với LABEL_CORRECTIONS):')
        for group in cross_groups:
            print('   ', group)

    df, label_log = apply_label_corrections(df)
    df, dedup_log = remove_same_label_duplicates(df)

    counts = df['label'].value_counts().reindex(CLASSES, fill_value=0)
    if (counts == 0).any():
        missing = counts[counts == 0].index.tolist()
        raise ValueError(f'Dữ liệu thiếu lớp: {missing}')
    print(f'[classes] Lệch lớp (max/min): {counts.max() / counts.min():.2f} lần')

    train_df, val_df, test_df = split_dataset(df, seed=seed)
    save_splits(train_df, val_df, test_df, csv_dir, label_log, dedup_log)
    return train_df, val_df, test_df


class ResizeWithPad:
    """Resize giữ tỉ lệ rồi thêm viền đen để tạo ảnh vuông."""

    def __init__(self, size, fill=0):
        if not isinstance(size, int) or size < 1:
            raise ValueError('Kích thước ảnh phải là số nguyên dương.')
        self.size = size
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        scale = self.size / max(width, height)
        new_w = max(1, round(width * scale))
        new_h = max(1, round(height * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new('RGB', (self.size, self.size), (self.fill,) * 3)
        canvas.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas

    def __repr__(self):
        return f'ResizeWithPad(size={self.size}, fill={self.fill})'


def load_rgb_image(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def build_resize(img_size):
    return ResizeWithPad(img_size)


@dataclass
class AugmentConfig:
    """Xác suất và mức biến đổi dùng cho tập train."""
    hflip_p: float = 0.5            # lật ngang
    vflip_p: float = 0.2            # lật dọc
    rotation_p: float = 0.5         # xoay
    rotation_degrees: float = 15.0  # góc xoay ngẫu nhiên trong [-15, +15] độ
    brightness_p: float = 0.5       # đổi độ sáng
    brightness_delta: float = 0.15  # hệ số sáng ngẫu nhiên trong [0.85, 1.15]
    contrast_p: float = 0.5         # đổi độ tương phản
    contrast_delta: float = 0.15    # hệ số tương phản ngẫu nhiên trong [0.85, 1.15]
    fill: int = 0                   # màu lấp vùng trống sau khi xoay (0 = đen, khớp với viền của ResizeWithPad)

    def __post_init__(self):
        for name in ('hflip_p', 'vflip_p', 'rotation_p', 'brightness_p', 'contrast_p'):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} phải nằm trong [0, 1], hiện là {value}.')
        if self.rotation_degrees < 0:
            raise ValueError('rotation_degrees phải >= 0.')
        for name in ('brightness_delta', 'contrast_delta'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} phải nằm trong [0, 1].')
        if not isinstance(self.fill, int) or not 0 <= self.fill <= 255:
            raise ValueError('fill phải là số nguyên trong [0, 255].')


def build_augmentation(config=None):
    """Mỗi phép biến đổi có xác suất áp dụng riêng."""
    cfg = config or AugmentConfig()
    return T.Compose([
        T.RandomHorizontalFlip(p=cfg.hflip_p),
        T.RandomVerticalFlip(p=cfg.vflip_p),
        T.RandomApply([T.RandomRotation(cfg.rotation_degrees, fill=cfg.fill, interpolation=T.InterpolationMode.BILINEAR)], p=cfg.rotation_p),
        T.RandomApply(
            [T.ColorJitter(brightness=(1 - cfg.brightness_delta, 1 + cfg.brightness_delta))],
            p=cfg.brightness_p),
        T.RandomApply(
            [T.ColorJitter(contrast=(1 - cfg.contrast_delta, 1 + cfg.contrast_delta))],
            p=cfg.contrast_p),
    ])


def build_postprocess(training, aug_config=None):
    steps = []
    if training:
        steps.append(build_augmentation(aug_config))
    steps.append(T.ToTensor())
    return T.Compose(steps)


class GarbageDataset(Dataset):
    """Đọc ảnh theo CSV. Có thể cache ảnh đã resize trong RAM."""

    def __init__(self, csv_path, img_size=IMG_SIZE, training=False, aug_config=None,
                 cache_resized=False, data_dir=None):
        self.data_dir = Path(data_dir or find_data_dir())
        self.resize = build_resize(img_size)
        self.postprocess = build_postprocess(training, aug_config)

        df = pd.read_csv(csv_path)
        if not {'filepath', 'label'}.issubset(df.columns) or df.empty:
            raise ValueError(f'{csv_path} phải có dữ liệu và 2 cột filepath, label.')
        if df['filepath'].isna().any() or not df['label'].isin(CLASSES).all():
            raise ValueError(f'{csv_path} có đường dẫn trống hoặc nhãn không hợp lệ.')

        self.filepaths = df['filepath'].tolist()
        self.labels = df['label'].map(CLASS_TO_INDEX).astype(int).tolist()

        self._cache = None
        if cache_resized:
            self._cache = [self._load_and_resize(i) for i in range(len(self.filepaths))]

    def _load_and_resize(self, index):
        return self.resize(load_rgb_image(self.data_dir / self.filepaths[index]))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        image = self._cache[index] if self._cache is not None else self._load_and_resize(index)
        return self.postprocess(image), self.labels[index]


def seed_everything(seed=SEED):
    """Đặt seed để hỗ trợ tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED, csv_dir=None,
                     aug_config=None, num_workers=2, cache_resized=False, data_dir=None):
    """Tạo DataLoader; chỉ tập train được shuffle và augmentation."""
    seed_everything(seed)
    data_dir = Path(data_dir) if data_dir is not None else find_data_dir()
    csv_dir = Path(csv_dir) if csv_dir else get_csv_dir(data_dir)

    common = dict(img_size=img_size, cache_resized=cache_resized, data_dir=data_dir)
    train_ds = GarbageDataset(csv_dir / 'train' / 'train.csv', training=True, aug_config=aug_config, **common)
    val_ds = GarbageDataset(csv_dir / 'val' / 'val.csv', training=False, **common)
    test_ds = GarbageDataset(csv_dir / 'test' / 'test.csv', training=False, **common)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


def compute_class_weights(train_csv_path=None):
    """Tính trọng số lớp từ tập train theo thứ tự CLASSES."""
    train_csv_path = Path(train_csv_path) if train_csv_path else get_csv_dir() / 'train' / 'train.csv'
    labels = pd.read_csv(train_csv_path)['label']
    if labels.empty or not labels.isin(CLASSES).all():
        raise ValueError('CSV train trống hoặc có nhãn không hợp lệ.')
    counts = labels.value_counts().reindex(CLASSES, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f'Tập train thiếu lớp: {counts[counts == 0].index.tolist()}')
    weights = len(labels) / (len(CLASSES) * counts.values)
    return torch.tensor(weights, dtype=torch.float32)


def show_batch(loader, n=6, title=''):
    import matplotlib.pyplot as plt

    if n < 1:
        raise ValueError('n phải lớn hơn hoặc bằng 1.')
    images, labels = next(iter(loader))
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3), squeeze=False)
    for ax, img, label in zip(axes[0], images[:n], labels[:n]):
        ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[int(label)], fontsize=8)
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_preprocessing()

    train_loader, val_loader, test_loader = make_dataloaders(num_workers=0)
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        images, labels = next(iter(loader))
        assert tuple(images.shape[1:]) == (3, IMG_SIZE, IMG_SIZE), images.shape
        assert images.dtype == torch.float32
        assert torch.isfinite(images).all()
        assert 0.0 <= images.min() and images.max() <= 1.0
        assert 0 <= labels.min() and labels.max() < len(CLASSES)
        print(f'[{name}] batch={tuple(images.shape)}, range=[{images.min():.3f}, {images.max():.3f}]')

    print('class_weights:', compute_class_weights())
