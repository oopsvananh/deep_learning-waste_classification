
"""Dự đoán ảnh rác bằng MobileNetV3 đã fine-tune."""

import argparse
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

from prepare_mobilenetv3 import DATA_DIR, WasteDataset, read_csv


# Kiến trúc MobileNetV3 đã fine-tune
class WasteMobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base = mobilenet_v3_large(weights=None)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base.classifier[0].in_features, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# Đánh giá mô hình trên tập test
def evaluate_test(model, classes, transform, device, data_dir, batch_size):
    rows = read_csv(data_dir / "test" / "test.csv")
    class_to_index = {name: i for i, name in enumerate(classes)}

    test_data = WasteDataset(
        rows, data_dir / "raw" / "garbage_classification", class_to_index
    )
    test_data.transform = transform

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    num_classes = len(classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss = 0

    # Dự đoán toàn bộ tập test
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total_loss += nn.functional.cross_entropy(
                outputs, labels, reduction="sum"
            ).item()

            for true, pred in zip(labels.cpu(), predictions.cpu()):
                confusion[true, pred] += 1

    # Tính các chỉ số đánh giá
    tp = confusion.diag().float()
    support = confusion.sum(dim=1)

    precision = tp / confusion.sum(dim=0).clamp(min=1)
    recall = tp / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    accuracy = tp.sum().item() / len(test_data)
    loss = total_loss / len(test_data)

    print("\nTest loss:", round(loss, 4))
    print("Test accuracy:", round(accuracy, 4))
    print("Macro precision:", round(precision.mean().item(), 4))
    print("Macro recall:", round(recall.mean().item(), 4))
    print("Macro F1:", round(f1.mean().item(), 4))

    print("\nKết quả từng lớp:")
    for i, name in enumerate(classes):
        print(
            f"{name}: Precision={precision[i]:.4f}, "
            f"Recall={recall[i]:.4f}, "
            f"F1={f1[i]:.4f}, Support={support[i]}"
        )


# Chọn ảnh hoặc đánh giá tập test
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=Path)
parser.add_argument("--test", action="store_true")
parser.add_argument("--weights", type=Path, default=Path("best_mobilenetv3.pth"))
parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--top-k", type=int, default=3)
args = parser.parse_args()

if args.image is None and not args.test:
    parser.error("Chọn --image hoặc --test.")

# Load checkpoint đã huấn luyện
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
classes = checkpoint["classes"]

model = WasteMobileNetV3(len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Tiền xử lý ảnh theo thông số lúc training
image_size = checkpoint["image_size"]

if isinstance(image_size, int):
    image_size = (image_size, image_size)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(checkpoint["mean"], checkpoint["std"])
])

print("Device:", device)

# Đánh giá trên tập test
if args.test:
    evaluate_test(
        model, classes, transform, device, args.data_dir, args.batch_size
    )

# Dự đoán một ảnh
else:
    image = Image.open(args.image).convert("RGB")
    X = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)[0]
        scores, indices = probabilities.topk(min(args.top_k, len(classes)))

    print("\nImage:", args.image)
    print("Prediction:", classes[indices[0].item()])

    for score, index in zip(scores.tolist(), indices.tolist()):
        print(f"{classes[index]}: {score:.2%}")
