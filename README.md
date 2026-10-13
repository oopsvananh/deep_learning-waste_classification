# Waste Classification

Bài tập phân loại ảnh rác thành 12 lớp bằng **Simple CNN**, **Complex CNN** và **MobileNetV3-Large fine-tuning**.

Dữ liệu: [Garbage Classification trên Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification). Ba model dùng cùng CSV chia tập; MobileNetV3 có cách chuẩn hóa và augmentation riêng.

## Cấu trúc thư mục

```text
data/
  raw/dataset_info.json
  raw/garbage_classification/        # Ảnh gốc theo từng lớp
  train/train.csv
  val/val.csv
  test/test.csv
preprocess/
  EDA.ipynb
  preprocessing.py
  show_after_preprocess.ipynb
  show_samples.py
models/
  model_1/
    simple.py
    test.py
  model_2/
    train_complex_cnn.py
    predict.py
  model_3/
    prepare_mobilenetv3.py
    train_mobilenetv3.py
    predict_mobilenetv3.py
weights/
  best_model1.pt
  complex_cnn_best_epoch187.pth
  best_mobilenetv3.pth
```

## Dữ liệu và tiền xử lý

Dataset gốc có 15.515 ảnh. Pipeline sửa ba nhãn đã xác định, loại ảnh trùng theo MD5 và chia tập bằng stratified sampling với seed 42. Việc sửa nhãn và loại ảnh trùng được thể hiện trong CSV, không sửa hoặc xóa ảnh gốc.

| Tập | Số ảnh | Tỉ lệ |
|---|---:|---:|
| Train | 10.847 | 70% |
| Validation | 2.324 | 15% |
| Test | 2.325 | 15% |
| Tổng sau làm sạch | 15.496 | 100% |

12 lớp: `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.

- **Simple CNN và Complex CNN:** chuyển RGB, giữ tỉ lệ và thêm padding đen bằng `ImageOps.pad` để đạt 224 × 224, đưa pixel về [0, 1].
- **MobileNetV3:** resize trực tiếp về 224 × 224, chuẩn hóa theo mean/std ImageNet.
- Augmentation chỉ áp dụng cho train. Hai CNN dùng lật, xoay và chỉnh độ sáng/tương phản; MobileNetV3 dùng cấu hình riêng trong `prepare_mobilenetv3.py`, có thêm thay đổi tỉ lệ.
- Class weights được tính từ nhãn train. Các CSV chia tập hiện được lưu trong Git.

Mở `preprocess/EDA.ipynb` để khám phá dữ liệu và `preprocess/show_after_preprocess.ipynb` để xem dữ liệu sau tiền xử lý. Nếu cần tạo lại ba CSV, chạy:

```powershell
python preprocess/preprocessing.py
```

Không cần tạo lại CSV chỉ để chạy predict.

## Cách chạy

Chạy từ thư mục gốc repository, bằng môi trường Python có `torch`, `torchvision`, `scikit-learn`, `pandas` và `Pillow`. Notebook cần thêm các thư viện được import trong từng notebook.

Trên Windows, nếu chưa kích hoạt môi trường ảo, thay `python` bằng `.\.venv\Scripts\python.exe`. Các script tự chọn CUDA nếu khả dụng, nếu không dùng CPU.

### 1. Simple CNN

```powershell
python models/model_1/simple.py
```

Model gồm ba khối Conv–BN–ReLU, Max/Average Pooling, Global Average Pooling, Dropout 0.3 và lớp phân loại 12 lớp. Huấn luyện tối đa 50 epoch bằng AdamW, learning rate 0.001, weighted cross-entropy và label smoothing 0.1. Model được chọn theo validation accuracy; early stopping có patience 5.

Script lưu `best_model1.pt` vào thư mục đang chạy, còn `test.py` đọc file tại `models/model_1/best_model1.pt`. Để đánh giá checkpoint có sẵn:

```powershell
Copy-Item weights/best_model1.pt models/model_1/best_model1.pt
python models/model_1/test.py
```

Nếu đánh giá model vừa train từ thư mục gốc, dùng file `best_model1.pt` ở thư mục gốc làm nguồn sao chép. `test.py` in accuracy, weighted precision/recall/F1, macro F1 và classification report từng lớp.

### 2. Complex CNN

```powershell
python models/model_2/train_complex_cnn.py
```

Complex CNN có **1.282.532 tham số**, được huấn luyện từ đầu:

```text
RGB input: 3 × 224 × 224
  → Ba nhánh Conv 3×3, 5×5, 7×7 + BN + ReLU
  → Ghép 6 + 5 + 5 = 16 kênh
  → Stage 1: 1 residual block, 16 kênh  → MaxPool → 112 × 112
  → Stage 2: 1 residual block, 32 kênh  → MaxPool → 56 × 56
  → Stage 3: 2 residual blocks, 64 kênh → MaxPool → 28 × 28
  → Stage 4: 1 residual block, 128 kênh          → 28 × 28
  → Global Average Pooling → Flatten
  → Linear 128 → 128 → ReLU → Dropout 0.5 → Linear 128 → 12
```

Stage 2–4 bắt đầu bằng Conv 3×3–BN–ReLU để tăng số kênh. Mỗi residual block dùng ba convolution 3×3 với số kênh `C → C → 2C → C`, giữ nguyên chiều cao và chiều rộng. Nhánh chính được cộng với đầu vào rồi qua ReLU. Lớp cuối trả về 12 logits; dự đoán là lớp có điểm cao nhất.

| Cấu hình | Giá trị |
|---|---|
| Epoch / batch size | 200 / 32 |
| Optimizer | Adam, learning rate ban đầu 1e-4 |
| Loss | Weighted cross-entropy |
| Scheduler | ReduceLROnPlateau theo validation loss; factor 0.5, patience 3, min LR 1e-6 |
| Chọn model | Validation loss thấp nhất |

Cấu hình được đặt trực tiếp trong code, không dùng tham số dòng lệnh. Validation chạy sau mỗi epoch. Trạng thái tốt nhất được giữ trong RAM, khôi phục sau 200 epoch và đánh giá test **một lần**.

Script in accuracy, macro precision, macro recall, macro F1, confusion matrix dạng số và tên file ảnh sai đầu tiên của mỗi lớp kèm nhãn thật/dự đoán. Lớp không có lỗi sẽ không in ví dụ. Script **không lưu checkpoint, log hoặc ảnh**.

Để đánh giá checkpoint có sẵn mà không train lại:

```powershell
python models/model_2/predict.py
```

`predict.py` nạp `weights/complex_cnn_best_epoch187.pth`, hỗ trợ tên khối đặc trưng cũ `features` và tên mới `feature_extraction`. Script in bốn chỉ số tổng hợp; không in confusion matrix hoặc ví dụ sai.

Kết quả đã kiểm tra với checkpoint epoch 187, 2.325 ảnh test và preprocessing hiện tại:

| Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---:|---:|---:|---:|
| 86.88% | 84.40% | 85.44% | 84.86% |

Các dòng `Precision`, `Recall` và `F1-score` của Complex CNN đều là macro average: mỗi lớp đóng góp ngang nhau. Khi đối chiếu kết quả, cần dùng cùng checkpoint, tập test và preprocessing.

### 3. MobileNetV3-Large fine-tuning

```powershell
python models/model_3/train_mobilenetv3.py
```

Model dùng trọng số ImageNet `IMAGENET1K_V1`, chỉ cập nhật năm lớp convolution cuối của backbone và classification head mới. BatchNorm trong backbone giữ nguyên running statistics. Head gồm GAP, Dropout 0.3, Linear 960 → 256, BatchNorm, ELU, Dropout 0.5 và Linear 256 → 12.

Huấn luyện bằng Adam với learning rate 1e-4, tối đa 50 epoch. Train dùng weighted cross-entropy và L1 (1e-5) trên trọng số hai lớp Linear. Validation/test trong script train dùng cross-entropy không có class weights, cộng L1. Model được chọn theo validation loss thấp nhất; early stopping dùng patience 10 và min delta 1e-4.

Script train giữ model tốt nhất trong RAM, đánh giá test ở cuối và in các chỉ số macro; không lưu checkpoint mới. Để đánh giá checkpoint có sẵn:

```powershell
python models/model_3/predict_mobilenetv3.py --test --weights weights/best_mobilenetv3.pth
```

Dự đoán một ảnh, thay đường dẫn ảnh cho phù hợp:

```powershell
python models/model_3/predict_mobilenetv3.py --image data/raw/garbage_classification/battery/battery1.jpg --weights weights/best_mobilenetv3.pth
```

## Đánh giá và so sánh

Validation được dùng để chọn model; test được dùng cho đánh giá cuối cùng. Khi lập bảng so sánh, dùng accuracy và các chỉ số **macro** nhất quán giữa các model. Không lấy weighted precision/recall/F1 của Simple CNN thay cho macro.

Ba model dùng cùng danh sách ảnh chia tập, nhưng khác kiến trúc, preprocessing và cách huấn luyện. Kết quả phản ánh toàn bộ cấu hình của từng model, không tách riêng tác dụng của từng thành phần.

## Nguồn dữ liệu

Nguồn dataset, ngày tải, số ảnh gốc theo lớp và SHA-256 của archive được lưu trong `data/raw/dataset_info.json`. Ảnh gốc được lưu trong Git; file ZIP và môi trường `.venv` được bỏ qua. Xem trang dataset để biết điều kiện sử dụng và quyền đối với ảnh.
