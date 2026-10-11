"""Đánh giá Complex CNN đã train trên toàn bộ tập test."""

from pathlib import Path

import torch
from torch import nn

from models.model_2.train_complex_cnn import (
    ComplexCNN,
    classification_metrics,
    evaluate,
)
from preprocess.preprocessing import (
    CLASSES,
    compute_class_weights,
    make_dataloaders,
)


# Path luôn được tính từ vị trí của predict.py nên dùng được trên Windows.
PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "weights" / "complex_cnn_best_epoch187.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset đã được chia sẵn; script chỉ sử dụng tập test.
    _, _, test_loader = make_dataloaders()

    model = ComplexCNN(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = compute_class_weights().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    test_result = evaluate(model, test_loader, loss_fn, device)
    metrics = classification_metrics(test_result)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test images     :", len(test_loader.dataset))
    print(f"Accuracy        : {metrics['accuracy']:.2%}")
    print(f"Precision       : {metrics['precision']:.2%}")
    print(f"Recall          : {metrics['recall']:.2%}")
    print(f"F1-score        : {metrics['f1_score']:.2%}")


if __name__ == "__main__":
    main()
