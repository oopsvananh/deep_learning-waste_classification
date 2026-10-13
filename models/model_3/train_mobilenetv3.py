
"""Fine-tune MobileNetV3-Large; chỉ in kết quả, không lưu log/checkpoint."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from prepare_mobilenetv3 import prepare, compute_class_weights


# 1. Settings
EPOCHS = 50
LR = 1e-4
L1 = 1e-5

PATIENCE = 10
MIN_DELTA = 1e-4


# 2. BatchNorm cho classifier
class HeadBatchNorm(nn.BatchNorm1d):
    def forward(self, x):
        # Batch cuối có thể chỉ còn một ảnh
        if self.training and x.shape[0] == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps
            )

        return super().forward(x)


# 3. Build model
class PartialMobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=weights)

        self.backbone = model.features
        # gọi toàn bộ từ đầu đến chỗ 960x7x7

        # Freeze toàn bộ backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Fine-tune 5 convolutional layers cuối
        conv_layers = [
            layer for layer in self.backbone.modules()
            if isinstance(layer, nn.Conv2d)
        ]
        # backbone.children() => chỉ trả về block
        # backbone.modules() => trả cả về block, cả về layers
        for layer in conv_layers[-5:]:
            for param in layer.parameters():
                param.requires_grad = True
        # tại sao chỉ fine tune 5 layers cuối (data ở mức trung bình; độ tương đồng cao với ImageNet)
        # tại sao chọn MobileNetV3-Large (model tân tiến) (lý thuyết: model này sinh ra là để deployment vì nó nhanh và chính xác cao; thực nghiệm: thực nghiệm trên 13models, model này thỏa mãn các mục tiêu)
        in_features = model.classifier[0].in_features
        # 960x7x7

        self.pool = nn.AdaptiveAvgPool2d(1)

        # 960x1x1

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            HeadBatchNorm(256),
            nn.ELU(),
            # ELU thì cái phần > 0 nó giống ReLu, <=0 thì nó sẽ là 1 cái hàm mũ 
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.train()

    def train(self, mode=True):
        super().train(mode)

        # Không cập nhật BatchNorm statistics của backbone
        self.backbone.eval()
        # Mình chỉ cập nhật ở các conv layers, nên nếu có batchnorm không cập nhật, giữ nguyên mean, STD ban đầu, không tính lại
        return self

    def forward(self, x):
        x = self.backbone(x)  # 960x7x7
        x = self.pool(x)  # 960x1x1
        x = x.flatten(1)  # 960
        x = self.head(x)  # 12

        return x

    def l1_penalty(self):
        penalty = (
            self.head[1].weight.abs().sum()
            + self.head[5].weight.abs().sum()
        )

        return L1 * penalty


# 4. Run one epoch
def run_epoch(model, loader, device, optimizer=None, confusion=None, class_weights=None):
    training = optimizer is not None

    model.train(training)

    total_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            # Phương thức model (images) trả ra cái vector cỡ 12

            loss = F.cross_entropy(outputs, labels, weight=class_weights)
            loss += model.l1_penalty()


            if training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Confusion matrix chỉ dùng khi test
            if confusion is not None:
                num_classes = confusion.size(0)

                confusion += torch.bincount(
                    (labels * num_classes + predictions).cpu(),
                    minlength=num_classes ** 2
                ).reshape(num_classes, num_classes)
            # ban đầu làm việc trên vector, sau đó reshape thành ma trận confusion => count từng batch một => xếp theo thứ tự tương ứng => trải thành ma trânj

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy


# 5. Train
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loaders, classes = prepare()

    # Class weights tính từ training set
    class_weights = compute_class_weights(
        loaders["train"].dataset, classes
    ).to(device)

    model = PartialMobileNet(
        num_classes=len(classes)
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), # Adam chỉ làm việc với các trọng số được cập nhật
        lr=LR
    )

    print("Device:", device)
    print("Classes:", len(classes))
    print("Class weights:", class_weights)

    best_loss = float("inf")
    stop_loss = float("inf")

    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = run_epoch(
            model,
            loaders["train"],
            device,
            optimizer,
            class_weights=class_weights
        )

        val_loss, val_acc = run_epoch(
            model,
            loaders["val"],
            device
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={train_loss:.4f}, acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
        )

        # Giữ model có validation loss thấp nhất trong RAM
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        # Early stopping
        if stop_loss - val_loss > MIN_DELTA:
            stop_loss = val_loss
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    # 6. Test
    model.load_state_dict(best_state)

    confusion = torch.zeros(
        len(classes),
        len(classes),
        dtype=torch.long
    )

    test_loss, test_acc = run_epoch(
        model,
        loaders["test"],
        device,
        confusion=confusion
    )

    true_positive = confusion.diag().float()

    precision = (
        true_positive
        / confusion.sum(dim=0).clamp(min=1)
    )

    recall = (
        true_positive
        / confusion.sum(dim=1).clamp(min=1)
    )

    f1 = (
        2 * precision * recall
        / (precision + recall).clamp(min=1e-12)
    )

    print()
    print("Best epoch:", best_epoch)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Macro precision: {precision.mean():.4f}")
    print(f"Macro recall: {recall.mean():.4f}")
    print(f"Macro F1: {f1.mean():.4f}")


if __name__ == "__main__":
    main()
