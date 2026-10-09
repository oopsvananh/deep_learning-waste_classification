"""Huấn luyện và đánh giá Complex CNN bằng PyTorch.

Chạy từ thư mục gốc của repository:
    python models/model_2/train_complex_cnn.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from preprocess.preprocessing import (  # noqa: E402
    CLASSES,
    compute_class_weights,
    make_dataloaders,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Complex CNN")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--test-frequency", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Thư mục chứa 12 thư mục class; mặc định dùng data/raw trong repo",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=PROJECT_DIR / "data",
        help="Thư mục chứa train/train.csv, val/val.csv và test/test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "complex_cnn",
    )
    return parser.parse_args()


class ParallelStem(nn.Module):
    """Nhìn ảnh ở ba vùng 3x3, 5x5 và 7x7 rồi nối các channels."""

    def __init__(self):
        super().__init__()
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )
        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )

    def forward(self, x):
        features = [
            self.branch_3x3(x),
            self.branch_5x5(x),
            self.branch_7x7(x),
        ]
        return torch.cat(features, dim=1)


class ResidualBlock(nn.Module):
    """Mở rộng kênh 2 lần, thu nhỏ rồi cộng với input."""

    def __init__(self, channels):
        super().__init__()
        expanded_channels = channels * 2

        self.spatial = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(
                channels,
                expanded_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(),
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.output_relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.spatial(x)
        x = self.expand(x)
        x = self.shrink(x)
        return self.output_relu(x + residual)


class ComplexCNN(nn.Module):
    """Bốn stage với năm residual block phân bố theo tỉ lệ 1-1-2-1."""

    def __init__(self, num_classes=12):
        super().__init__()

        self.features = nn.Sequential(
            ParallelStem(),
            ResidualBlock(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ),
            ResidualBlock(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            ResidualBlock(128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Học một epoch: dự đoán, tính loss, rồi cập nhật model."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)              # 1. Model dự đoán
        loss = loss_fn(logits, labels)       # 2. So sánh với nhãn đúng
        loss.backward()                     # 3. Tính gradient
        optimizer.step()                    # 4. Cập nhật model

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_images += batch_size

    return {
        "loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Đánh giá model; không tính gradient và không cập nhật weights."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_images += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
        "labels": all_labels,
        "predictions": all_predictions,
    }


def classification_metrics(result):
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        result["labels"],
        result["predictions"],
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": result["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def save_checkpoint(path, model, epoch, val_loss):
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    if args.test_frequency < 1:
        raise ValueError("--test-frequency phải lớn hơn hoặc bằng 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")
    print(f"Kích thước ảnh: {args.image_size} | Batch size: {args.batch_size}")

    train_loader, val_loader, test_loader = make_dataloaders(
        img_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        csv_dir=args.csv_dir,
    )

    model = ComplexCNN(num_classes=len(CLASSES)).to(device)
    train_csv = args.csv_dir / "train" / "train.csv"
    class_weights = compute_class_weights(train_csv).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Số tham số: {parameter_count:,}")

    checkpoint_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "training_history.csv"
    test_history_path = args.output_dir / "test_metrics.csv"
    best_val_loss = float("inf")
    history = []
    test_history = []

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device)
        val_result = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_result["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            save_checkpoint(checkpoint_path, model, epoch, best_val_loss)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train {row['train_accuracy']:.4f} | "
            f"val {row['val_accuracy']:.4f} | "
            f"val_loss {row['val_loss']:.4f}"
        )

        if epoch % args.test_frequency == 0:
            test_result = evaluate(model, test_loader, loss_fn, device)
            test_row = {"epoch": epoch, **classification_metrics(test_result)}
            test_history.append(test_row)
            pd.DataFrame(test_history).to_csv(test_history_path, index=False)
            print(
                f"Test @ epoch {epoch}: "
                f"Accuracy={test_row['accuracy']:.2%} | "
                f"Precision={test_row['precision']:.2%} | "
                f"Recall={test_row['recall']:.2%} | "
                f"F1-score={test_row['f1_score']:.2%}"
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_test = evaluate(model, test_loader, loss_fn, device)
    final_metrics = classification_metrics(final_test)
    final_report = classification_report(
        final_test["labels"],
        final_test["predictions"],
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        **final_metrics,
        "classification_report": final_report,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        confusion_matrix(
            final_test["labels"],
            final_test["predictions"],
            labels=list(range(len(CLASSES))),
        ),
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(args.output_dir / "confusion_matrix.csv")

    print(
        f"Checkpoint tốt nhất: epoch {checkpoint['epoch']} | "
        f"Accuracy={final_metrics['accuracy']:.2%} | "
        f"Precision={final_metrics['precision']:.2%} | "
        f"Recall={final_metrics['recall']:.2%} | "
        f"F1-score={final_metrics['f1_score']:.2%}"
    )
    print(f"Đã lưu kết quả vào: {args.output_dir}")


if __name__ == "__main__":
    main()
