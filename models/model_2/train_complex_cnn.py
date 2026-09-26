"""Huấn luyện và đánh giá Complex CNN để phân loại rác.

Chạy từ thư mục gốc của repository:
    python models/model_2/train_complex_cnn.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# Giúp import data_pipeline.py dù script được gọi từ thư mục nào.
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "preprocess"))


def parse_args():
    """Đọc các cài đặt thường cần đổi giữa những lần chạy."""
    parser = argparse.ArgumentParser(description="Train the Complex CNN")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-frequency", type=int, default=10,
                        help="Chạy validation sau mỗi N epoch")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size của một process trên một GPU")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-mb", type=int, default=8192,
                        help="Giới hạn VRAM của một process")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_DIR / "artifacts" / "complex_cnn")
    return parser.parse_args()


def build_model(img_size, num_classes, learning_rate):
    """Tạo Complex CNN theo đúng kiến trúc trong README."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size, img_size, 3)),

        # Khối 1: nhìn vùng rộng, tinh chỉnh chi tiết rồi giảm kích thước ảnh.
        tf.keras.layers.Conv2D(32, 5, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),

        # Khối 2 học thêm đặc trưng từ đầu ra nhỏ hơn của khối 1.
        tf.keras.layers.Conv2D(64, 5, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),

        # Khối 3 học đặc trưng cấp cao để dùng khi phân loại.
        tf.keras.layers.Conv2D(128, 5, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),

        # Lấy trung bình mỗi kênh thay vì nối tất cả giá trị bằng Flatten.
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="complex_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def collect_labels(dataset):
    """Lấy nhãn theo đúng thứ tự ảnh mà model.predict() đã dự đoán."""
    return np.concatenate([labels.numpy() for _, labels in dataset])


class TestMetricsEveryNEpochs(tf.keras.callbacks.Callback):
    """Đo accuracy và F1 trên test set sau mỗi N epoch."""

    def __init__(self, test_dataset, every_n_epochs, output_path):
        super().__init__()
        self.test_dataset = test_dataset
        self.true_labels = collect_labels(test_dataset)
        self.every_n_epochs = every_n_epochs
        self.output_path = output_path
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        epoch_number = epoch + 1
        if epoch_number % self.every_n_epochs != 0:
            return

        probabilities = self.model.predict(self.test_dataset, verbose=0)
        predictions = probabilities.argmax(axis=1)
        record = {
            "epoch": epoch_number,
            "test_accuracy": accuracy_score(self.true_labels, predictions),
            "test_macro_f1": f1_score(
                self.true_labels, predictions, average="macro", zero_division=0
            ),
            "test_weighted_f1": f1_score(
                self.true_labels, predictions, average="weighted", zero_division=0
            ),
        }
        self.records.append(record)
        pd.DataFrame(self.records).to_csv(self.output_path, index=False)
        print(
            f"\nTest @ epoch {epoch_number}: "
            f"accuracy={record['test_accuracy']:.4f}, "
            f"macro_f1={record['test_macro_f1']:.4f}, "
            f"weighted_f1={record['test_weighted_f1']:.4f}"
        )


def configure_device(memory_limit_mb):
    """Chỉ cho phép một GPU và giới hạn VRAM để tránh chiếm toàn bộ card."""
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        raise RuntimeError(
            "Mỗi process chỉ được nhìn thấy một GPU. "
            "Hãy đặt CUDA_VISIBLE_DEVICES về đúng một GPU."
        )
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
        )
        print(f"Dùng 1 GPU, giới hạn VRAM: {memory_limit_mb} MB")
    else:
        print("Không tìm thấy GPU; dùng CPU")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_device(args.gpu_memory_mb)
    tf.keras.utils.set_random_seed(args.seed)

    # Import sau configure_device để TensorFlow chưa khởi tạo GPU quá sớm.
    from data_pipeline import CLASSES, make_custom_cnn_datasets

    img_size = 128
    csv_dir = PROJECT_DIR / "data"

    print(f"Batch size: {args.batch_size}")

    train_ds, val_ds, test_ds, class_weights = make_custom_cnn_datasets(
        img_size=img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        csv_dir=csv_dir,
    )

    model = build_model(img_size, len(CLASSES), args.learning_rate)

    model.summary()

    callbacks = [
        # Chỉ giữ model ở epoch có validation loss thấp nhất.
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_history.csv"),
        TestMetricsEveryNEpochs(
            test_dataset=test_ds,
            every_n_epochs=args.validation_frequency,
            output_path=args.output_dir / "test_metrics_every_10_epochs.csv",
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_freq=args.validation_frequency,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Nạp lại checkpoint tốt nhất trước khi đánh giá test.
    model = tf.keras.models.load_model(args.output_dir / "best_model.keras")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    probabilities = model.predict(test_ds)
    predictions = probabilities.argmax(axis=1)
    true_labels = collect_labels(test_ds)

    report_text = classification_report(
        true_labels,
        predictions,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        true_labels,
        predictions,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print("\nTest metrics:", test_metrics)
    print("\nClassification report:\n", report_text)

    results = {
        "epochs_completed": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "classification_report": report_dict,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (args.output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    pd.DataFrame(
        confusion_matrix(true_labels, predictions),
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(args.output_dir / "confusion_matrix.csv")

    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
