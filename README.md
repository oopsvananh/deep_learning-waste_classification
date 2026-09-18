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
│   └── TrashNet_EDA.ipynb           # EDA cũ, cần chuyển đổi trước khi dùng dataset mới
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
- `preprocess/TrashNet_EDA.ipynb` được giữ lại làm tài liệu EDA cũ. Notebook này vẫn dùng đường dẫn, 6 nhãn và kết quả của TrashNet; cần cập nhật trước khi chạy với dataset mới.
- Thêm bước làm sạch và chia dữ liệu trong `preprocess/`. Lưu dữ liệu đã xử lý vào `data/processed/`, các tập chia vào `data/train/`, `data/val/`, `data/test/`, mỗi tập chứa các thư mục class.
- Dùng cùng một cách chia dữ liệu cho cả 3 mô hình để so sánh kết quả. Chỉ áp dụng augmentation cho tập train; dùng validation để chọn mô hình và test để đánh giá cuối cùng.
- Thay tên `model_1`, `model_2`, `model_3` khi chọn kiến trúc cụ thể; đặt phần định nghĩa và cấu hình riêng của từng mô hình trong thư mục tương ứng.

Hiện chỉ có dữ liệu gốc và notebook EDA. Các thư mục còn lại là khung chuẩn bị, chưa chia dữ liệu hoặc triển khai huấn luyện. Dữ liệu sinh ra trong `processed/train/val/test` được bỏ qua bởi Git.

## Nguồn và phạm vi sử dụng

- Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- Tác giả: Mostafa Mohamed. Ngày tải: 18/09/2026.
- Nhãn lấy trực tiếp từ thư mục dataset: `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.
- Xem `data/raw/dataset_info.json` để biết số ảnh thực tế và SHA-256 của archive tải về. Dữ liệu raw chưa được loại ảnh trùng, sửa nhãn hoặc chia tập.
- Kaggle ghi giấy phép: **Database: Open Database, Contents: © Original Authors**. Quyền đối với ảnh thuộc tác giả gốc; xem mô tả nguồn về sử dụng cho nghiên cứu và phi lợi nhuận.
- Dữ liệu TrashNet cũ đã được thay thế trong phiên bản hiện tại; lịch sử Git vẫn giữ các phiên bản trước.
