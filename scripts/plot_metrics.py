"""Plot train loss and validation mask AP from one run metrics CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_metrics(metrics_path: Path, output: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"metrics file is empty: {metrics_path}")
    epochs = [int(row["epoch"]) for row in rows]
    losses = [float(row["loss_total"]) for row in rows]
    mask_ap = [float(row["valid_mask_map"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, mask_ap, marker="o")
    axes[1].set(title="Validation mask AP", xlabel="Epoch", ylabel="mask_map")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot an instance segmentation run metrics.csv")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(plot_metrics(args.metrics, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
