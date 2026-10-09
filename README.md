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
│   ├── EDA.ipynb                  # Khám phá dataset 12 lớp
│   ├── Preprocessing.ipynb        # Chạy và minh họa tiền xử lý
│   ├── preprocessing.py           # Chia dữ liệu và tạo PyTorch DataLoader
│   └── show_batch.ipynb           # Xem một batch sau tiền xử lý
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
- Mở `preprocess/EDA.ipynb`, chọn Python kernel và **Run All**. Notebook tự tìm dataset từ workspace hoặc repository.
- EDA kiểm kê ảnh, đối chiếu metadata lúc tải, phân tích mất cân bằng lớp, ảnh mẫu, kích thước/định dạng, độ sáng/tương phản/chi tiết và ảnh trùng/gần trùng. Kết quả chỉ hiển thị trực tiếp trong notebook, không xuất file riêng.
- Xem nhóm ảnh trùng/gần trùng trực tiếp bằng `REVIEW_START` và `REVIEW_LIMIT`; đặt `REVIEW_LIMIT = None` để xem tất cả. Nhóm pHash cần được kiểm tra thủ công.
- Notebook EDA chỉ đọc ảnh gốc, không xóa/sửa nhãn hoặc chia dữ liệu.
- Chạy `preprocess/Preprocessing.ipynb` để kiểm tra ảnh, sửa các nhãn đã xác nhận, loại ảnh trùng và tạo CSV cho train/validation/test.
- Dùng cùng một cách chia dữ liệu cho cả 3 mô hình để so sánh kết quả. Chỉ áp dụng augmentation cho tập train; dùng validation để chọn mô hình và test để đánh giá cuối cùng.
- Thay tên `model_1`, `model_2`, `model_3` khi chọn kiến trúc cụ thể; đặt phần định nghĩa và cấu hình riêng của từng mô hình trong thư mục tương ứng.

Pipeline tiền xử lý PyTorch nằm trong `preprocess/preprocessing.py`. Complex CNN
nằm trong `models/model_2/train_complex_cnn.py`. Dữ liệu sinh ra trong
`processed/train/val/test` và kết quả trong `artifacts/` được bỏ qua bởi Git.

## Thiết kế Complex CNN

Complex CNN vẫn sử dụng các thành phần cơ bản đã có trong Simple CNN như
`Conv2d`, `BatchNorm2d`, ReLU và `MaxPool2d`. Model chỉ dùng convolution thông
thường. Stem có ba convolution `3x3`, `5x5` và `7x7` chạy song song; phần còn
lại dùng kernel `3x3`. Model không dùng depthwise convolution, pointwise
convolution hoặc convolution stride 2.

Mỗi residual block giữ nguyên kích thước đầu vào và đầu ra. Ở giữa block, số
kênh được mở rộng gấp đôi để học thêm đặc trưng rồi thu nhỏ về số kênh ban đầu
để cộng với input. MaxPooling tiếp tục đảm nhiệm việc giảm chiều cao và chiều
rộng giữa các stage. Model có bốn stage và năm residual block, phân bố
`1-1-2-1`. Cách phân bố này giữ stage thứ ba sâu hơn, bám theo tỉ lệ macro
`3-3-9-3` của ConvNeXt nhưng được thu nhỏ để phù hợp bài tập. Số kênh tăng dần
`16-32-64-128` để giữ model gọn hơn cho tập train 10,847 ảnh.

Kiến trúc:

```text
Input 224x224x3

  -> Parallel stem
       Conv 3x3: 3 -> 6 -> BatchNorm -> ReLU
       Conv 5x5: 3 -> 5 -> BatchNorm -> ReLU
       Conv 7x7: 3 -> 5 -> BatchNorm -> ReLU
       Concatenate channels: 6 + 5 + 5 = 16

  -> Residual block 16
       Conv 3x3: 16 -> 16  -> BatchNorm -> ReLU
       Conv 3x3: 16 -> 32  -> BatchNorm -> ReLU
       Conv 3x3: 32 -> 16  -> BatchNorm
       Add input -> ReLU
  -> MaxPooling

  -> Conv 3x3: 16 -> 32 -> BatchNorm -> ReLU
  -> Residual block 32
       Conv 3x3: 32 -> 32  -> BatchNorm -> ReLU
       Conv 3x3: 32 -> 64  -> BatchNorm -> ReLU
       Conv 3x3: 64 -> 32  -> BatchNorm
       Add input -> ReLU
  -> MaxPooling

  -> Conv 3x3: 32 -> 64 -> BatchNorm -> ReLU
  -> Residual block 64 (lặp 2 lần)
       Conv 3x3: 64  -> 64  -> BatchNorm -> ReLU
       Conv 3x3: 64  -> 128 -> BatchNorm -> ReLU
       Conv 3x3: 128 -> 64  -> BatchNorm
       Add input -> ReLU
  -> MaxPooling

  -> Conv 3x3: 64 -> 128 -> BatchNorm -> ReLU
  -> Residual block 128
       Conv 3x3: 128 -> 128 -> BatchNorm -> ReLU
       Conv 3x3: 128 -> 256 -> BatchNorm -> ReLU
       Conv 3x3: 256 -> 128 -> BatchNorm
       Add input -> ReLU

  -> AdaptiveAvgPool2d
  -> Linear 128 -> ReLU -> Dropout
  -> Linear 12
```

Lớp cuối trả về 12 logits. Không đặt Softmax trong model vì
`CrossEntropyLoss` của PyTorch đã xử lý logits trực tiếp; lớp có logit lớn nhất
vẫn là dự đoán của model.

### Parallel stem

Ba convolution đầu tiên cùng nhận một ảnh. Kernel `3x3` tập trung vào chi tiết
nhỏ, `5x5` nhìn vùng trung bình và `7x7` nhìn vùng rộng hơn. Mỗi nhánh dùng
padding phù hợp nên đều trả về feature map `224x224`; model nối các nhánh theo
chiều channels để tạo 16 channels cho stage đầu tiên.

```python
branch_3x3 = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(6),
    nn.ReLU(),
)
branch_5x5 = nn.Sequential(
    nn.Conv2d(3, 5, kernel_size=5, padding=2, bias=False),
    nn.BatchNorm2d(5),
    nn.ReLU(),
)
branch_7x7 = nn.Sequential(
    nn.Conv2d(3, 5, kernel_size=7, padding=3, bias=False),
    nn.BatchNorm2d(5),
    nn.ReLU(),
)

output = torch.cat(
    [branch_3x3(x), branch_5x5(x), branch_7x7(x)],
    dim=1,
)
```

### Mở rộng và thu nhỏ số kênh

Ví dụ, block 32 kênh biến đổi `32 -> 64 -> 32`. Lớp giữa có nhiều kênh hơn để
học các đặc trưng trung gian khác nhau. Lớp cuối thu nhỏ về 32 kênh để output có
cùng shape với input. Hệ số mở rộng 2 lần giúp model có thêm kênh trung gian
nhưng vẫn giữ số tham số và phép tính ở mức hợp lý.

### Residual connection

Đầu ra của nhánh convolution được cộng với input ban đầu:

```text
output = input + transformed_input
```

Nhánh convolution học phần thông tin cần bổ sung, còn input có một đường truyền
trực tiếp đến output. Hai tensor chỉ được cộng khi có cùng chiều cao, chiều rộng
và số kênh; vì vậy mỗi block phải thu nhỏ về số kênh ban đầu trước phép cộng.
Phép cộng được dùng thay cho concatenate để không làm tăng số kênh của output.

### Global average pooling

Sau các khối convolution, dữ liệu có dạng `H x W x C`, trong đó `C` là số kênh
đặc trưng. Global average pooling lấy trung bình toàn bộ `H x W` của từng kênh,
biến mỗi kênh thành một con số:

```text
H x W x C -> 1 x 1 x C -> vector C phần tử
```

Ví dụ, đầu vào `28x28x128` sẽ trở thành vector có 128 phần tử. Mỗi phần tử tóm
tắt mức độ xuất hiện của một đặc trưng trên toàn ảnh. Vector này được đưa vào
lớp Linear để phân loại.

Cách này tránh nối trực tiếp toàn bộ `28x28x128 = 100,352` giá trị vào classifier.
Global average pooling chỉ tạo 128 giá trị, nên phần phân loại cần ít tham số
hơn và giảm nguy cơ ghi nhớ quá chi tiết vị trí của ảnh train.

Complex CNN được so sánh với Simple CNN trên cùng train/validation/test split,
augmentation và class weight. Validation được dùng để chọn checkpoint; test
không được dùng để chọn model.

Code PyTorch để định nghĩa, huấn luyện và đánh giá model nằm tại
`models/model_2/train_complex_cnn.py`. Chạy từ thư mục gốc của repository:

```sh
python models/model_2/train_complex_cnn.py
```

Mỗi process dùng một GPU trong `CUDA_VISIBLE_DEVICES`; batch size mặc định của
mỗi model là 32.

Script lưu model tốt nhất, lịch sử train, classification report và confusion
matrix vào `artifacts/complex_cnn/`.

Mặc định model train đủ 200 epoch. Validation chạy sau mỗi epoch và chỉ
`val_loss` được dùng để lưu `best_model.pt`. Test cũng được đo sau mỗi epoch và
in bốn metric: Accuracy, Macro Precision, Macro Recall và Macro F1-score. Macro
average tính mỗi lớp ngang nhau, phù hợp khi số ảnh giữa các lớp không cân bằng.
Cuối cùng, checkpoint có validation loss thấp nhất được nạp lại để đánh giá
test.

Chạy từ thư mục gốc của repository, thay GPU `0` nếu cần:

```sh
CUDA_VISIBLE_DEVICES=0 python models/model_2/train_complex_cnn.py \
  --epochs 200 --test-frequency 1 --batch-size 32 --image-size 224 \
  --learning-rate 1e-4 \
  --output-dir artifacts/complex_cnn
```

### Kết quả phiên bản stem 5x5 trước đó

Hai seed được train độc lập trong 200 epoch. Checkpoint tốt nhất được chọn chỉ
bằng validation loss; test không được dùng để chọn model.

| Seed | Best epoch | Test accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|---:|
| 42 | 89 | 87.91% | 85.99% | 87.94% |
| 43 | 191 | 86.32% | 83.95% | 86.30% |

Phiên bản cũ có `1,282,460` tham số. Kết quả seed 42 tại epoch 89 là kết quả chính vì
checkpoint này có validation loss tốt nhất trong run seed 42 và đạt kết quả
cao hơn seed 43 trên cả ba test metric.

### Kết quả Parallel stem hiện tại

Parallel stem có `1,282,532` tham số và được train 200 epoch với seed 42, ảnh
`224x224`, batch size 32 và learning rate `1e-4`. Checkpoint epoch 187 được chọn
chỉ bằng validation loss, sau đó mới dùng để tính kết quả test chính thức.

| Seed | Epoch được chọn | Test accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 187 | 87.18% | 85.25% | 87.19% |

Full training log gồm train, validation và test metrics của toàn bộ 200 epoch
nằm tại `artifacts/logs/complex_cnn_parallel_stem_seed42.log`.

## Nguồn và phạm vi sử dụng

- Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- Tác giả: Mostafa Mohamed. Ngày tải: 18/09/2026.
- Nhãn lấy trực tiếp từ thư mục dataset: `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.
- Xem `data/raw/dataset_info.json` để biết số ảnh thực tế và SHA-256 của archive tải về. Dữ liệu raw chưa được loại ảnh trùng, sửa nhãn hoặc chia tập.
- Kaggle ghi giấy phép: **Database: Open Database, Contents: © Original Authors**. Quyền đối với ảnh thuộc tác giả gốc; xem mô tả nguồn về sử dụng cho nghiên cứu và phi lợi nhuận.
- Dữ liệu TrashNet cũ đã được thay thế trong phiên bản hiện tại; lịch sử Git vẫn giữ các phiên bản trước.
