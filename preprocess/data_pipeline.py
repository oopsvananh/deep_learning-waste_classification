"""
data_pipeline.py — Biến ảnh Garbage Classification thành tf.data.Dataset để train model.

MỤC LỤC (đọc theo đúng thứ tự này nếu muốn hiểu toàn bộ file):
    1. Cấu hình chung + tìm thư mục ảnh gốc
    2. Xử lý 1 ảnh: đọc file -> ép về RGB -> resize
    3. Augmentation (chỉ dùng cho tập train)
    4. Class weight (xử lý lệch lớp)
    5. Hàm lõi make_dataset(): ghép 4 bước trên thành 1 pipeline hoàn chỉnh
    6. Hàm tiện ích cho Simple CNN / Complex CNN

===============================================================================
 CÁCH DÙNG NHANH
===============================================================================

Bạn làm Simple CNN hoặc Complex CNN (2 người dùng chung 1 hàm):

    from data_pipeline import make_custom_cnn_datasets

    train_ds, val_ds, test_ds, class_weight_dict = make_custom_cnn_datasets()
    model.fit(train_ds, validation_data=val_ds, class_weight=class_weight_dict, epochs=...)

Bạn làm Transfer learning + fine-tune — tự chọn backbone và tự import đúng
`preprocess_input` của backbone đó, rồi gọi thẳng hàm lõi `make_dataset` (mục 5):

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from data_pipeline import make_dataset, compute_class_weights, DATA_DIR

    CSV_DIR = DATA_DIR.parent.parent
    IMG_SIZE = 224   # đổi theo đúng size backbone bạn chọn yêu cầu

    train_ds = make_dataset(CSV_DIR / 'train' / 'train.csv', training=True,
                             img_size=IMG_SIZE, batch_size=32, seed=42, normalize_fn=preprocess_input)
    val_ds = make_dataset(CSV_DIR / 'val' / 'val.csv', training=False,
                           img_size=IMG_SIZE, batch_size=32, seed=42, normalize_fn=preprocess_input)
    test_ds = make_dataset(CSV_DIR / 'test' / 'test.csv', training=False,
                            img_size=IMG_SIZE, batch_size=32, seed=42, normalize_fn=preprocess_input)
    class_weight_dict = compute_class_weights(CSV_DIR / 'train' / 'train.csv')

    # Đổi backbone khác thì đổi đúng import, ví dụ:
    # from tensorflow.keras.applications.resnet50 import preprocess_input
    # from tensorflow.keras.applications.efficientnet import preprocess_input

    LƯU Ý QUAN TRỌNG: normalize_fn nhận ảnh [0,255] và phải phù hợp với model pretrained.
    Không chia 255 trước preprocess_input hoặc chuẩn hóa lần nữa nếu pipeline đã xử lý.
    Kiểm tra tài liệu của model vì preprocessing có thể được tích hợp trong model.
    Dùng sai preprocessing có thể làm giảm chất lượng dù chương trình vẫn chạy.

Backbone là model pretrained do người train tự chọn, không phải tham số của pipeline.

File train.csv/val.csv/test.csv là kết quả xuất ra từ notebook Preprocessing.ipynb (Bước 5).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


# ===============================================================================
# 1. CẤU HÌNH CHUNG + TÌM THƯ MỤC ẢNH GỐC
# ===============================================================================

SEED = 42
IMG_SIZE = 128       # kích thước ảnh mặc định cho Simple/Complex CNN
BATCH_SIZE = 32

CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes',
    'trash', 'white-glass',
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}

# train.csv/val.csv/test.csv chỉ ghi đường dẫn TƯƠNG ĐỐI (vd 'metal/metal459.jpg'),
# nên cần tìm DATA_DIR để ghép thành đường dẫn đầy đủ khi đọc ảnh.
def _find_data_dir():
    search_bases = [Path(__file__).resolve().parent.parent, Path.cwd(), *Path.cwd().parents]
    relative_paths = [
        'data/raw/garbage_classification',
        'deep_learning-waste_classification/data/raw/garbage_classification',
    ]
    for base in search_bases:
        for relative in relative_paths:
            candidate = base / relative
            if candidate.is_dir():
                return candidate.resolve()
    return None


DATA_DIR = _find_data_dir()
if DATA_DIR is None:
    raise FileNotFoundError(
        'Không tìm thấy thư mục dữ liệu garbage_classification. '
        'Nếu notebook làm model nằm ở vị trí khác, sửa lại _find_data_dir() cho đúng.'
    )


# ===============================================================================
# 2. XỬ LÝ 1 ẢNH: đọc file -> ép về RGB -> resize
# ===============================================================================
# CHÚ Ý: hàm này CHƯA chuẩn hóa pixel (chưa chia 255) — ảnh trả về vẫn ở thang [0, 255].
# Lý do: mỗi model cần 1 kiểu chuẩn hóa khác nhau (CNN tự thiết kế dùng [0,1], mỗi
# backbone transfer learning lại cần 1 công thức riêng) — nên bước chuẩn hóa được
# tách ra làm riêng ở mục 5, không gộp vào đây.

def load_image(filepath, label, img_size):
    full_path = tf.strings.join([str(DATA_DIR), filepath], separator='/')
    raw_bytes = tf.io.read_file(full_path)

    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)  # ép về 3 kênh RGB
    image.set_shape([None, None, 3])  # để TensorFlow biết chắc số kênh, tránh lỗi shape ở bước resize

    image = tf.image.resize_with_pad(  # giữ tỉ lệ khung hình, thêm viền đen thay vì kéo méo ảnh
        image, target_height=img_size, target_width=img_size, method='bilinear', antialias=True,
    )
    image = tf.clip_by_value(image, 0.0, 255.0)  # nội suy có thể tạo sai số nhỏ vượt biên [0, 255]

    return image, label


# ===============================================================================
# 3. AUGMENTATION — chỉ dùng cho tập train
# ===============================================================================
# Vẫn hoạt động trên thang [0, 255] (giống ảnh ra từ load_image), vì chuẩn hóa
# chưa xảy ra ở bước này — xem lý do ở mục 2.

_augment_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05),                              # xoay ngẫu nhiên khoảng ±18 độ
    tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 255.0)),   # đổi độ sáng ngẫu nhiên ±10%
    tf.keras.layers.RandomContrast(0.1),                                # đổi độ tương phản ngẫu nhiên ±10%
], name='augmentation')


def augment_image(image, label):
    image = _augment_layers(image, training=True)
    image = tf.clip_by_value(image, 0.0, 255.0)  # augmentation cũng có thể đẩy nhẹ pixel ra ngoài biên
    return image, label


# ===============================================================================
# 4. CLASS WEIGHT — xử lý lệch lớp
# ===============================================================================

def compute_class_weights(train_csv_path):
    """Đọc train.csv, trả về {class_index: weight} để truyền vào model.fit(class_weight=...)."""
    train_df = pd.read_csv(train_csv_path)
    weights = compute_class_weight(class_weight='balanced', classes=np.array(CLASSES), y=train_df['label'])
    return {CLASS_TO_INDEX[name]: weight for name, weight in zip(CLASSES, weights)}


# ===============================================================================
# 5. HÀM LÕI: ghép mục 2 + 3 + chuẩn hóa thành 1 pipeline hoàn chỉnh
# ===============================================================================
# Simple/Complex CNN dùng hàm tiện ích ở mục 6.
# Transfer learning gọi trực tiếp hàm này với normalize_fn phù hợp model pretrained.

def make_dataset(csv_path, training, img_size, batch_size, seed, normalize_fn):
    """
    Đọc 1 file CSV (cột 'filepath', 'label') và trả về 1 tf.data.Dataset đã sẵn sàng
    đưa vào model.fit(). Thứ tự xử lý:

        đọc CSV -> xáo trộn (nếu train) -> load_image (mục 2)
                -> augment_image (chỉ nếu train, mục 3) -> normalize_fn (chuẩn hóa cuối cùng)
                -> gộp batch -> prefetch

    normalize_fn: hàm nhận ảnh thang [0,255], trả về ảnh đã chuẩn hóa đúng model cần.
    """
    split_df = pd.read_csv(csv_path)
    if not {'filepath', 'label'}.issubset(split_df.columns) or split_df.empty:
        raise ValueError(f'{csv_path} phải có dữ liệu và 2 cột filepath, label.')
    if split_df['filepath'].isna().any() or not split_df['label'].isin(CLASSES).all():
        raise ValueError(f'{csv_path} có đường dẫn trống hoặc nhãn không hợp lệ.')

    filepaths = split_df['filepath'].values
    labels = split_df['label'].map(CLASS_TO_INDEX).values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=len(split_df), seed=seed)

    dataset = dataset.map(
        lambda filepath, label: load_image(filepath, label, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(
        lambda image, label: (normalize_fn(image), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ===============================================================================
# 6. HÀM TIỆN ÍCH CHO SIMPLE CNN / COMPLEX CNN
# ===============================================================================
# Bạn làm Transfer learning KHÔNG dùng hàm này — xem hướng dẫn "CÁCH DÙNG NHANH"
# ở đầu file để gọi thẳng make_dataset() với đúng preprocess_input backbone bạn chọn.

def normalize_to_unit_range(image):
    """Chuẩn hóa cho CNN tự thiết kế: đưa pixel từ [0, 255] về [0, 1]."""
    return image / 255.0


def make_custom_cnn_datasets(img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED, csv_dir=None):
    """
    Dùng cho Simple CNN và Complex CNN tự thiết kế.
    Trả về (train_ds, val_ds, test_ds, class_weight_dict), pixel đã chuẩn hóa về [0, 1].
    """
    csv_dir = csv_dir or DATA_DIR.parent.parent

    train_ds = make_dataset(csv_dir / 'train' / 'train.csv', training=True, img_size=img_size,
                             batch_size=batch_size, seed=seed, normalize_fn=normalize_to_unit_range)
    val_ds = make_dataset(csv_dir / 'val' / 'val.csv', training=False, img_size=img_size,
                           batch_size=batch_size, seed=seed, normalize_fn=normalize_to_unit_range)
    test_ds = make_dataset(csv_dir / 'test' / 'test.csv', training=False, img_size=img_size,
                            batch_size=batch_size, seed=seed, normalize_fn=normalize_to_unit_range)
    class_weight_dict = compute_class_weights(csv_dir / 'train' / 'train.csv')

    return train_ds, val_ds, test_ds, class_weight_dict


# ===============================================================================
# Tự kiểm tra nhanh: python data_pipeline.py
# ===============================================================================
if __name__ == '__main__':
    train_ds, val_ds, test_ds, class_weight_dict = make_custom_cnn_datasets()
    images, labels = next(iter(train_ds))
    print(f'[custom CNN] shape={images.shape}, range=[{images.numpy().min():.3f}, {images.numpy().max():.3f}]')
    print('class_weight_dict:', class_weight_dict)
