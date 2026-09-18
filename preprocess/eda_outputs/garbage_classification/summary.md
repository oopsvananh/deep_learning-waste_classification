# Garbage Classification — EDA


### Kết quả định lượng
- Kiểm kê **15,515** ảnh, đọc được **15,515**, lỗi **0**, cảnh báo **0**.
- **12 lớp**; tỷ lệ lớn nhất/nhỏ nhất có ảnh **8.77 lần**. Baseline luôn đoán lớp đông nhất trên toàn bộ dữ liệu: **34.32%**.
- **892** kích thước ảnh lưu trữ; chế độ màu: **P, RGB**.
- **18 nhóm trùng byte**, **18 nhóm trùng pixel**, trong đó **3 nhóm trùng pixel khác nhãn**.
- Nếu mỗi nhóm pixel giống hệt chỉ giữ một ảnh thì có **19 ảnh dư**; notebook chưa xóa ảnh.
- **285 cặp pHash** trong ngưỡng Hamming ≤ 4, bao gồm ảnh trùng pixel.
- Gom được **162 nhóm cần xem lại / 368 ảnh**, trong đó **17 nhóm nhiều nhãn**.

### Ý nghĩa đối với bước tiếp theo
- Báo cáo macro-F1, recall từng lớp và confusion matrix; cân nhắc class weights chỉ dựa trên tập train sau khi chia.
- Kiểm tra các nhóm khác nhãn trước; ảnh gần trùng là ứng viên, không phải kết luận nhãn sai.
- Khi chia tập, giữ ảnh cùng nhóm đã xác nhận trong cùng một tập để tránh rò rỉ. Các nhóm pHash có thể có false positive và không bao phủ mọi ảnh cùng vật thể.
- Tách đánh giá trên ảnh thực tế nếu cần kiểm chứng khả năng tổng quát hóa. EDA này không đánh giá mô hình.
- CSV, hình và phiên bản thư viện đã lưu trong `preprocess/eda_outputs/garbage_classification/`.
