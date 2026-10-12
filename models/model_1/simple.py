import sys
import time
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.preprocessing import make_dataloaders, compute_class_weights, CLASSES

EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5
BATCH_SIZE = 32

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Thiết bị:', device)

    train_loader, val_loader, test_loader = make_dataloaders()
    class_weights = compute_class_weights().to(device)

    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),                                                # 224 -> 112
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AvgPool2d(2),                                                # 112 -> 56
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),                                                # 56 -> 28
        nn.AdaptiveAvgPool2d(1),                                        # 28x28 -> 1x1
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(128, len(CLASSES)),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Số tham số: {parameter_count:,}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0
    no_improve = 0
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss, train_correct, train_n = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_n += len(labels)

        # --- validation ---
        model.eval()
        val_loss, val_correct, val_n = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * len(labels)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_n += len(labels)
        val_acc = val_correct / val_n
        scheduler.step(val_loss / val_n)

        print(f'Epoch {epoch:02d} | train {train_loss/train_n:.3f}/{train_correct/train_n:.3f} '
              f'| val {val_loss/val_n:.3f}/{val_acc:.3f}')

        # ghi lại model khi val acc tăng, dừng sớm nếu không tăng thêm
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, 'best_model1.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'Dừng sớm tại epoch {epoch} (val acc không tăng {PATIENCE} epoch)')
                break

    print(f'\nTổng thời gian chạy: {(time.time() - start) / 60:.1f} phút')
    print(f'Model tốt nhất theo val: {best_val_acc:.4f} (đã lưu vào best_model1.pt)')

if __name__ == "__main__":
    main()