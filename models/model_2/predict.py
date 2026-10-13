"""Đánh giá Complex CNN đã train trên toàn bộ tập test."""

import sys
from pathlib import Path

import torch
from torch import nn


# Thêm thư mục gốc để Python tìm được model và preprocessing trên Windows.
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from models.model_2.train_complex_cnn import (  # noqa: E402
    ComplexCNN,
    classification_metrics,
    evaluate,
)
from preprocess.preprocessing import (  # noqa: E402
    CLASSES,
    compute_class_weights,
    make_dataloaders,
)


CHECKPOINT_PATH = PROJECT_DIR / "weights" / "complex_cnn_best_epoch187.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset đã được chia sẵn; script chỉ sử dụng tập test.
    _, _, test_loader = make_dataloaders()

    model = ComplexCNN(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    # Support checkpoints saved with the old name "features".
    model_state = {}
    for name, value in checkpoint["model_state_dict"].items():
        if name.startswith("features."):
            name = name.replace("features.", "feature_extraction.", 1)
        model_state[name] = value
    model.load_state_dict(model_state)

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
