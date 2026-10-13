
"""Đánh giá MobileNetV3 đã fine-tune trên tập test."""

import torch
from pathlib import Path

from train_mobilenetv3 import PartialMobileNet, run_epoch
from prepare_mobilenetv3 import prepare


# Load dữ liệu và checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaders, classes = prepare()

checkpoint_path = Path(__file__).resolve().parents[2] / "weights" / "best_mobilenetv3.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

model = PartialMobileNet(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Đánh giá tập test
confusion = torch.zeros(len(classes), len(classes), dtype=torch.long)

with torch.inference_mode():
    loss, accuracy = run_epoch(
        model, loaders["test"], device, confusion=confusion
    )

tp = confusion.diag().float()
precision = tp / confusion.sum(0).clamp(min=1)
recall = tp / confusion.sum(1).clamp(min=1)
f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

print("Device:", device)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Macro precision: {precision.mean():.4f}")
print(f"Macro recall: {recall.mean():.4f}")
print(f"Macro F1: {f1.mean():.4f}")

print("\nKết quả từng lớp:")
for i, name in enumerate(classes):
    print(
        f"{name}: Precision={precision[i]:.4f}, "
        f"Recall={recall[i]:.4f}, F1={f1[i]:.4f}, "
        f"Support={int(confusion[i].sum())}"
    )
