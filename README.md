# Waste Classification

Phân loại rác với [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) của Mostafa Mohamed, tổ chức cho 3 mô hình dùng chung dữ liệu và quy trình đánh giá.

```text
deep_learning-waste_classification/
├── data/
│   ├── raw/
│   │   ├── dataset_info.json        # Nguồn, checksum archive và số ảnh từng lớp
│   │   └── garbage_classification/  # Dữ liệu gốc theo 12 class
│   ├── processed/                   # Dữ liệu sau tiền xử lý
│   ├── train/                       # Tập huấn luyện
│   ├── val/                         # Tập validation
│   └── test/                        # Tập kiểm thử cuối cùng
├── preprocess/
│   └── Garbage_Classification_EDA.ipynb # EDA cho dataset 12 lớp hiện tại
├── train/                           # Code huấn luyện dùng chung
├── val/                             # Code validation, chọn cấu hình/checkpoint
├── test/                            # Code đánh giá cuối cùng
└── models/
    ├── model_1/                     # Kiến trúc và cấu hình mô hình 1
    ├── model_2/                     # Kiến trúc và cấu hình mô hình 2
    └── model_3/                     # Kiến trúc và cấu hình mô hình 3
```

## Cách sử dụng

- Dữ liệu mới nằm trong `data/raw/garbage_classification/`, mỗi thư mục con là một nhãn. Ảnh đã giải nén được lưu trong Git; file ZIP được bỏ qua.
- Mở `preprocess/Garbage_Classification_EDA.ipynb`, chọn Python kernel và **Run All**. Thư viện cần có: `numpy`, `pandas`, `matplotlib`, `Pillow`, `scipy`, `IPython`. Notebook tự tìm dataset từ workspace, repo hoặc thư mục `preprocess`; có thể đặt `DATA_DIR` thủ công.
- EDA kiểm kê ảnh, đối chiếu metadata lúc tải, phân tích mất cân bằng lớp, ảnh mẫu, kích thước/định dạng, độ sáng/tương phản/chi tiết và ảnh trùng/gần trùng. Kết quả chỉ hiển thị trực tiếp trong notebook, không xuất file riêng.
- Xem nhóm ảnh trùng/gần trùng trực tiếp bằng `REVIEW_START` và `REVIEW_LIMIT`; đặt `REVIEW_LIMIT = None` để xem tất cả. Nhóm pHash cần được kiểm tra thủ công.
- Notebook EDA chỉ đọc ảnh gốc, không xóa/sửa nhãn hoặc chia dữ liệu. Notebook `TrashNet_EDA.ipynb` cũ tại máy được giữ nguyên để tham khảo, không dùng cho dataset hiện tại.
- Thêm bước làm sạch và chia dữ liệu trong `preprocess/`. Lưu dữ liệu đã xử lý vào `data/processed/`, các tập chia vào `data/train/`, `data/val/`, `data/test/`, mỗi tập chứa các thư mục class.
- Dùng cùng một cách chia dữ liệu cho cả 3 mô hình để so sánh kết quả. Chỉ áp dụng augmentation cho tập train; dùng validation để chọn mô hình và test để đánh giá cuối cùng.
- Thay tên `model_1`, `model_2`, `model_3` khi chọn kiến trúc cụ thể; đặt phần định nghĩa và cấu hình riêng của từng mô hình trong thư mục tương ứng.

Hiện chỉ có dữ liệu gốc và notebook EDA. Các thư mục còn lại là khung chuẩn bị, chưa chia dữ liệu hoặc triển khai huấn luyện. Dữ liệu sinh ra trong `processed/train/val/test` được bỏ qua bởi Git.

## Thiết kế Complex CNN

Complex CNN vẫn sử dụng các thành phần cơ bản đã có trong Simple CNN như
`Conv2D`, `BatchNormalization` và `MaxPooling2D`. Điểm khác biệt chính là mỗi
khối có hai lớp convolution để học đặc trưng kỹ hơn trước khi pooling, đồng
thời sử dụng ReLU, kernel `5x5` và global average pooling.

Kiến trúc dự kiến:

```text
Input 128x128x3
  -> Conv 32, 5x5 -> BatchNorm -> ReLU
  -> Conv 32, 3x3 -> BatchNorm -> ReLU
  -> MaxPooling

  -> Conv 64, 5x5 -> BatchNorm -> ReLU
  -> Conv 64, 3x3 -> BatchNorm -> ReLU
  -> MaxPooling

  -> Conv 128, 5x5 -> BatchNorm -> ReLU
  -> Conv 128, 3x3 -> BatchNorm -> ReLU
  -> MaxPooling

  -> GlobalAveragePooling
  -> Dense 128 -> ReLU -> Dropout
  -> Dense 12 -> Softmax
```

### ReLU

ReLU là hàm kích hoạt đơn giản: giữ nguyên giá trị dương và đổi giá trị âm thành
`0`. Trong model này, ReLU được đặt sau Batch Normalization:

```python
layers.Conv2D(32, 5, padding='same', use_bias=False),
layers.BatchNormalization(),
layers.Activation('relu'),
```

### Kernel lớn hơn

Simple CNN dùng kernel `3x3`: tại mỗi vị trí, convolution quan sát một vùng gồm
9 phần tử của đầu vào. Complex CNN thêm kernel `5x5`, nên tại mỗi vị trí lớp có
thể quan sát một vùng rộng hơn gồm 25 phần tử. Vùng quan sát rộng hơn giúp model
kết hợp các chi tiết nằm xa nhau hơn, chẳng hạn hình dáng tổng thể, đường viền
hoặc một phần lớn hơn của vật thể.

Kernel `5x5` cần nhiều phép tính hơn `3x3`, vì vậy mỗi khối chỉ dùng `5x5` ở
convolution đầu tiên để thu thập thông tin trên vùng rộng. Convolution `3x3`
tiếp theo tiếp tục tinh chỉnh các đặc trưng cục bộ mà không làm model quá nặng.

### Global average pooling

Sau các khối convolution, dữ liệu có dạng `H x W x C`, trong đó `C` là số kênh
đặc trưng. Global average pooling lấy trung bình toàn bộ `H x W` của từng kênh,
biến mỗi kênh thành một con số:

```text
H x W x C -> 1 x 1 x C -> vector C phần tử
```

Ví dụ, đầu vào `16x16x128` sẽ trở thành vector có 128 phần tử. Mỗi phần tử tóm
tắt mức độ xuất hiện của một đặc trưng trên toàn ảnh. Vector này được đưa vào
lớp Dense để phân loại.

Cách này thay cho `Flatten`. `Flatten` nối toàn bộ `16x16x128 = 32,768` giá trị
thành một vector lớn, trong khi global average pooling chỉ tạo 128 giá trị. Vì
vậy phần phân loại cần ít tham số hơn, model nhẹ hơn và giảm nguy cơ ghi nhớ quá
chi tiết vị trí của ảnh train.

Complex CNN được so sánh với Simple CNN trên cùng train/validation/test split,
augmentation, class weight và seed. Validation được dùng để chọn checkpoint;
test được đo định kỳ để theo dõi accuracy và F1.

Code huấn luyện và đánh giá nằm tại
`models/model_2/train_complex_cnn.py`. Chạy từ thư mục gốc của repository:

```sh
python models/model_2/train_complex_cnn.py
```

Mỗi process chỉ dùng một GPU trong `CUDA_VISIBLE_DEVICES` và mặc định giới hạn
VRAM ở 8 GB. Muốn dùng hai GPU thì chạy hai process độc lập với hai seed và hai
thư mục output khác nhau; script sẽ dừng với thông báo lỗi nếu một process vô
tình nhìn thấy nhiều GPU. Cách này giúp hai lần train độc lập, không cần đồng bộ
giữa các GPU.

Script lưu model tốt nhất, lịch sử train, classification report và confusion
matrix vào `artifacts/complex_cnn/`.

Mặc định model train đủ 200 epoch. Sau mỗi 10 epoch, script chạy validation để
chọn checkpoint và chạy test để ghi `Test accuracy`, `Macro-F1` và
`Weighted-F1` vào `test_metrics_every_10_epochs.csv`. Model có `val_loss` thấp
nhất được lưu làm checkpoint tốt nhất và được nạp lại để đánh giá cuối cùng.

### Chạy lại kết quả 86.24%

Chạy từ thư mục gốc của repository (thay GPU `0` nếu cần):

```sh
CUDA_VISIBLE_DEVICES=0 python models/model_2/train_complex_cnn.py \
  --epochs 200 --validation-frequency 10 --batch-size 32 \
  --learning-rate 1e-4 --seed 42 --gpu-memory-mb 8192 \
  --output-dir artifacts/complex_cnn_relu_seed42
```

Kết quả tốt nhất đã ghi nhận tại epoch 170: **Test accuracy 86.24%**,
**Macro-F1 83.84%**, **Weighted-F1 86.32%**. Xem lại dòng này bằng:

```sh
awk -F, 'NR == 1 || $1 == 170' \
  artifacts/complex_cnn_relu_seed42/test_metrics_every_10_epochs.csv
```

## Nguồn và phạm vi sử dụng

- Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- Tác giả: Mostafa Mohamed. Ngày tải: 18/09/2026.
- Nhãn lấy trực tiếp từ thư mục dataset: `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.
- Xem `data/raw/dataset_info.json` để biết số ảnh thực tế và SHA-256 của archive tải về. Dữ liệu raw chưa được loại ảnh trùng, sửa nhãn hoặc chia tập.
- Kaggle ghi giấy phép: **Database: Open Database, Contents: © Original Authors**. Quyền đối với ảnh thuộc tác giả gốc; xem mô tả nguồn về sử dụng cho nghiên cứu và phi lợi nhuận.
- Dữ liệu TrashNet cũ đã được thay thế trong phiên bản hiện tại; lịch sử Git vẫn giữ các phiên bản trước.
