import matplotlib.pyplot as plt

from preprocessing import make_dataloaders, CLASSES


# vẽ n ảnh đầu tiên của 1 batch
def show_batch(loader, n=6, title=''):
    images, labels = next(iter(loader))
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(CLASSES[int(labels[i])], fontsize=8)
        axes[0, i].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_loader, val_loader, test_loader = make_dataloaders()
    show_batch(train_loader, title='Train (có augmentation)')
    show_batch(val_loader, title='Val (chỉ resize)')