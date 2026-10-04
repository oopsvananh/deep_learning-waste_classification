"""Ghi tiến độ của các lần train vào file TXT.

Ví dụ:
    python models/model_2/monitor_training_progress.py \
        --run seed42=artifacts/complex_cnn_seed42 \
        --run seed43=artifacts/complex_cnn_seed43
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Save training progress to TXT")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run theo định dạng ten=thu_muc_output",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--test-frequency", type=int, default=1)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training_progress.txt"),
    )
    return parser.parse_args()


def parse_runs(values):
    runs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Run không đúng định dạng ten=thu_muc: {value}")
        name, directory = value.split("=", maxsplit=1)
        runs.append((name, Path(directory)))
    return runs


def read_last_csv_row(path):
    if not path.exists():
        return None

    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows[-1] if rows else None


def make_bar(current, total, width=40):
    completed = round(width * current / total)
    return "[" + "#" * completed + "-" * (width - completed) + "]"


def format_run(name, run_dir, total_epochs, test_frequency):
    row = read_last_csv_row(run_dir / "training_history.csv")
    if row is None:
        return f"{name}: waiting for the first epoch"

    epoch = int(row["epoch"])
    percent = 100 * epoch / total_epochs
    lines = [
        f"{name}: {make_bar(epoch, total_epochs)} "
        f"{epoch}/{total_epochs} ({percent:.1f}%)",
        f"  train_accuracy={float(row['train_accuracy']):.4f}  "
        f"val_accuracy={float(row['val_accuracy']):.4f}  "
        f"val_loss={float(row['val_loss']):.4f}",
    ]

    test_row = read_last_csv_row(run_dir / "test_metrics.csv")
    next_test_epoch = min(
        (epoch // test_frequency + 1) * test_frequency,
        total_epochs,
    )
    if test_row is None:
        lines.append(f"  test: waiting; next_test_epoch={next_test_epoch}")
    else:
        lines.append(
            f"  test@{int(test_row['epoch'])}: "
            f"accuracy={float(test_row['test_accuracy']):.4f}  "
            f"macro_f1={float(test_row['test_macro_f1']):.4f}  "
            f"weighted_f1={float(test_row['test_weighted_f1']):.4f}  "
            f"next_test_epoch={next_test_epoch}"
        )
    return "\n".join(lines)


def write_text_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def main():
    args = parse_args()
    runs = parse_runs(args.run)

    while True:
        sections = [
            f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}"
        ]
        finished = True

        for name, run_dir in runs:
            row = read_last_csv_row(run_dir / "training_history.csv")
            if row is None or int(row["epoch"]) < args.epochs:
                finished = False
            progress = format_run(
                name,
                run_dir,
                args.epochs,
                args.test_frequency,
            )
            sections.append(progress)
            write_text_atomic(run_dir / "progress.txt", progress + "\n")

        write_text_atomic(args.output, "\n\n".join(sections) + "\n")
        if finished or args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
