# Chạy từ thư mục gốc: python models/model_2/train_complex_cnn.py

# 1. Import thư viện
# import json
import sys
import time
from pathlib import Path

# import pandas as pd
import torch
from sklearn.metrics import (
    # classification_report,
    # confusion_matrix,
    precision_recall_fscore_support
)
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from preprocess.preprocessing import (
    BATCH_SIZE,
    CLASSES,
    IMG_SIZE,
    compute_class_weights,
    make_dataloaders
)


# 2. Cấu hình huấn luyện
EPOCHS = 200
LEARNING_RATE = 1e-4
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "complex_cnn"


# 3. Các khối CNN
class ParallelStem(nn.Module):
    def __init__(self):
        super().__init__()

        # Ba nhánh tích chập, ghép thành 16 kênh
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )

        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )

    def forward(self, x):
        features = [
            self.branch_3x3(x),
            self.branch_5x5(x),
            self.branch_7x7(x)
        ]

        return torch.cat(features, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        expanded_channels = channels * 2

        # Giữ số kênh -> tăng gấp đôi -> giảm về số kênh ban đầu
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.expand = nn.Sequential(
            nn.Conv2d(
                channels, expanded_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU()
        )

        self.shrink = nn.Sequential(
            nn.Conv2d(
                expanded_channels, channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels)
        )

        self.output_relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.spatial(x)
        x = self.expand(x)
        x = self.shrink(x)

        # Cộng lại đầu vào của block
        return self.output_relu(x + residual)


# 4. Xây dựng mô hình
class ComplexCNN(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()

        self.features = nn.Sequential(
            # Đầu vào đa nhánh: 3 x 224 x 224 -> 16 x 224 x 224
            ParallelStem(),

            # Stage 1: 16 x 224 x 224 -> 16 x 112 x 112
            ResidualBlock(16),
            nn.MaxPool2d(kernel_size=2),

            # Stage 2: 16 -> 32 kênh, kích thước 112 -> 56
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            ResidualBlock(32),
            nn.MaxPool2d(kernel_size=2),

            # Stage 3: 32 -> 64 kênh, kích thước 56 -> 28
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2),

            # Stage 4: 64 -> 128 kênh, kích thước giữ 28 x 28
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            ResidualBlock(128)
        )

        # Tổng hợp đặc trưng và phân loại
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)

        return self.classifier(x)


# 5. Huấn luyện một epoch
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_images = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        batch_loss = loss_fn(outputs, labels)

        batch_loss.backward()
        optimizer.step()

        predicted_labels = outputs.argmax(dim=1)
        images_in_batch = labels.size(0)

        # Cộng dồn loss và số ảnh dự đoán đúng
        train_loss_sum += batch_loss.item() * images_in_batch
        correct_in_batch = (predicted_labels == labels).sum().item()
        train_correct += correct_in_batch
        train_images += images_in_batch

    train_loss = train_loss_sum / train_images
    train_accuracy = train_correct / train_images

    return train_loss, train_accuracy


# 6. Đánh giá validation và test
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Đánh giá model; không tính gradient và không cập nhật weights."""
    model.eval()  # Tắt Dropout, dùng thống kê BatchNorm đã học

    total_loss = 0.0
    total_correct = 0
    total_images = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        batch_loss = loss_fn(outputs, labels)
        predicted_labels = outputs.argmax(dim=1)

        images_in_batch = labels.size(0)
        total_loss += batch_loss.item() * images_in_batch
        total_correct += (predicted_labels == labels).sum().item()
        total_images += images_in_batch

        # Giữ nhãn để tính các chỉ số phân loại
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predicted_labels.cpu().tolist())

    return {
        "loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# 7. Tính các chỉ số phân loại
def classification_metrics(result):
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        result["labels"],
        result["predictions"],
        labels=list(range(len(CLASSES))),
        average="macro",  # Mỗi lớp đóng góp ngang nhau
        zero_division=0
    )

    return {
        "accuracy": result["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


# 8. Nạp dữ liệu
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Thiết bị: {device}")
    print(f"Kích thước ảnh: {IMG_SIZE} | Batch size: {BATCH_SIZE}")

    # Dùng các tập train/val/test đã chia sẵn trong CSV
    train_loader, val_loader, test_loader = make_dataloaders()

    # 9. Khởi tạo model, loss và optimizer
    model = ComplexCNN(num_classes=len(CLASSES)).to(device)

    # Trọng số lớp để xử lý mất cân bằng số mẫu
    class_weights = compute_class_weights().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Số tham số: {parameter_count:,}")

    # 10. Huấn luyện
    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    # history_path = OUTPUT_DIR / "training_history.csv"

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    # history = []

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        val_result = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_result["loss"]
        val_accuracy = val_result["accuracy"]

        # Giảm learning rate khi validation loss không cải thiện
        scheduler.step(val_loss)
        # current_learning_rate = optimizer.param_groups[0]["lr"]

        # Lưu kết quả mỗi epoch
        # epoch_result = {
        #     "epoch": epoch,
        #     "train_loss": train_loss,
        #     "train_accuracy": train_accuracy,
        #     "val_loss": val_loss,
        #     "val_accuracy": val_accuracy,
        #     "learning_rate": current_learning_rate,
        # }
        # history.append(epoch_result)
        # history_table = pd.DataFrame(history)
        # history_table.to_csv(history_path, index=False)

        # Giữ model có validation loss thấp nhất trong RAM
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train {train_accuracy:.4f} | "
            f"val {val_accuracy:.4f} | "
            f"val_loss {val_loss:.4f}"
        )

    training_seconds = time.perf_counter() - training_start

    # 11. Test model tốt nhất
    model.load_state_dict(best_state)

    # Lưu model tốt nhất để có thể nạp lại và predict mà không cần train lại
    checkpoint = {
        "epoch": best_epoch,
        "val_loss": best_val_loss,
        "model_state_dict": best_state,
        "classes": CLASSES,
        "image_size": IMG_SIZE
    }
    torch.save(checkpoint, checkpoint_path)

    final_test = evaluate(model, test_loader, loss_fn, device)
    final_metrics = classification_metrics(final_test)

    # true_labels = final_test["labels"]
    # predicted_labels = final_test["predictions"]

    # Báo cáo từng loại rác
    # final_report = classification_report(
    #     true_labels,
    #     predicted_labels,
    #     labels=list(range(len(CLASSES))),
    #     target_names=CLASSES,
    #     output_dict=True,
    #     zero_division=0
    # )

    # 12. Lưu kết quả
    # results = {
    #     "best_epoch": checkpoint["epoch"],
    #     "best_val_loss": checkpoint["val_loss"],
    #     "accuracy": final_metrics["accuracy"],
    #     "precision": final_metrics["precision"],
    #     "recall": final_metrics["recall"],
    #     "f1_score": final_metrics["f1_score"],
    #     "classification_report": final_report,
    # }
    # with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as file:
    #     json.dump(results, file, indent=2)

    # Ma trận nhầm lẫn: hàng là nhãn thật, cột là dự đoán
    # matrix = confusion_matrix(
    #     true_labels,
    #     predicted_labels,
    #     labels=list(range(len(CLASSES)))
    # )
    # matrix_table = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
    # matrix_table.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # 13. In kết quả
    hours, remaining_seconds = divmod(training_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)

    print("\nKết quả của model tốt nhất:")
    print("Epoch    :", best_epoch)
    print(f"Accuracy : {final_metrics['accuracy']:.2%}")
    print(f"Precision: {final_metrics['precision']:.2%}")
    print(f"Recall   : {final_metrics['recall']:.2%}")
    print(f"F1-score : {final_metrics['f1_score']:.2%}")
    print(f"Thời gian train: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print("Đã lưu model:", checkpoint_path)


# Cần giữ khi chạy DataLoader nhiều worker trên Windows
if __name__ == "__main__":
    main()
