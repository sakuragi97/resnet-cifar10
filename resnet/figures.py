"""Make the figures used in the README.

    python -m resnet.figures --checkpoint runs/resnet20/model.pt --runs runs --out assets

Each figure is skipped when its inputs (checkpoint, history.csv files) are missing.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
from torchvision import datasets

from resnet.data import CLASSES, build_dataset, build_loader, denormalize
from resnet.engine import predict
from resnet.evaluate import confusion_matrix
from resnet.utils import load_checkpoint, resolve_device

BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, BAD = "#0ca30c", "#d03b3b"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def read_history(path: Path) -> dict[str, list[float]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def plot_samples(data_dir: str, out: Path, per_class: int = 10) -> None:
    """A grid of raw test images, one row per class."""
    dataset = datasets.CIFAR10(data_dir, train=False, download=True)
    picks = {c: [] for c in range(len(CLASSES))}
    for image, label in dataset:
        if len(picks[label]) < per_class:
            picks[label].append(image)
        if all(len(images) == per_class for images in picks.values()):
            break

    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 0.9, len(CLASSES) * 0.95))
    for label, images in picks.items():
        for col, image in enumerate(images):
            ax = axes[label, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[label, 0].set_ylabel(CLASSES[label], rotation=0, ha="right", va="center", color=INK, fontsize=10)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(out)
    plt.close(fig)


def plot_history(history: dict[str, list[float]], title: str, out: Path) -> None:
    """Loss and error curves for one training run."""
    epochs = history["epoch"]
    fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax_loss.plot(epochs, history["train_loss"], color=BLUE, label="train")
    ax_loss.plot(epochs, history["test_loss"], color=ORANGE, label="test")
    ax_loss.set(title="Cross-entropy loss", xlabel="epoch")

    train_err = [100 * (1 - a) for a in history["train_acc"]]
    test_err = [100 * (1 - a) for a in history["test_acc"]]
    ax_err.plot(epochs, train_err, color=BLUE, label="train")
    ax_err.plot(epochs, test_err, color=ORANGE, label="test")
    ax_err.set(title="Error (%)", xlabel="epoch")
    ax_err.annotate(
        f"{test_err[-1]:.2f}% test error",
        xy=(epochs[-1], test_err[-1]),
        xytext=(-8, 18),
        textcoords="offset points",
        ha="right",
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK_SOFT, "lw": 0.8},
    )

    for ax in (ax_loss, ax_err):
        ax.legend(loc="upper right")
        ax.set_xlim(1, epochs[-1])
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(cm: torch.Tensor, out: Path) -> None:
    """Row-normalized confusion matrix: each row sums to 100%."""
    percent = 100 * cm / cm.sum(dim=1, keepdim=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(percent, cmap=BLUES, vmin=0, vmax=100)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = percent[i, j].item()
            if round(value) >= 1:
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value > 50 else INK,
                )
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="predicted class", ylabel="true class")
    ax.set_title("Confusion matrix on the test set (% of each true class)", loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_predictions(images, targets, logits, out: Path, rows: int = 3, cols: int = 8) -> None:
    """Test images with the predicted class and its softmax probability."""
    probs = logits.softmax(dim=1)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.75))
    for ax, image, target, p in zip(axes.flat, images, targets, probs, strict=False):
        pred = p.argmax().item()
        correct = pred == target.item()
        ax.imshow(denormalize(image).permute(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GOOD if correct else BAD)
            spine.set_linewidth(2.5)
        label = f"{CLASSES[pred]} {p[pred]:.0%}"
        if not correct:
            label += f"\ntrue: {CLASSES[target]}"
        ax.set_title(("✓ " if correct else "✗ ") + label, fontsize=8.5, color=INK, fontweight="normal")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_degradation(runs: dict[str, dict[str, list[float]]], out: Path) -> None:
    """Training (dashed) and test (solid) error of plain vs residual networks, as in Fig. 6 of the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (kind, title) in zip(axes, [("plain", "Plain networks"), ("resnet", "ResNets")], strict=True):
        for depth, color in [(20, BLUE), (56, ORANGE)]:
            history = runs.get(f"{kind}{depth}")
            if history is None:
                continue
            epochs = history["epoch"]
            ax.plot(
                epochs,
                [100 * (1 - a) for a in history["train_acc"]],
                "--",
                color=color,
                lw=1.6,
                label=f"{depth} layers, train",
            )
            ax.plot(
                epochs,
                [100 * (1 - a) for a in history["test_acc"]],
                color=color,
                label=f"{depth} layers, test",
            )
        ax.set(title=title, xlabel="epoch")
        ax.set_xlim(1, None)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("error (%)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the README figures.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/resnet20/model.pt"))
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="folder with <name>/history.csv")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    plot_samples(args.data_dir, args.out / "cifar10_samples.png")
    print(f"wrote {args.out / 'cifar10_samples.png'}")

    histories = {p.parent.name: read_history(p) for p in sorted(args.runs.glob("*/history.csv"))}
    if "resnet20" in histories:
        plot_history(histories["resnet20"], "ResNet-20 on CIFAR-10", args.out / "training_curves.png")
        print(f"wrote {args.out / 'training_curves.png'}")
    if {"plain20", "plain56", "resnet20", "resnet56"} <= histories.keys():
        plot_degradation(histories, args.out / "plain_vs_residual.png")
        print(f"wrote {args.out / 'plain_vs_residual.png'}")

    if args.checkpoint.exists():
        device = resolve_device(args.device)
        model, _ = load_checkpoint(args.checkpoint, device)
        loader = build_loader(build_dataset(args.data_dir, train=False), 256, train=False, num_workers=2)
        logits, targets = predict(model, loader, device)

        cm = confusion_matrix(logits.argmax(dim=1), targets, len(CLASSES))
        plot_confusion(cm, args.out / "confusion_matrix.png")
        print(f"wrote {args.out / 'confusion_matrix.png'}")

        # A fixed random sample of the test set, so the figure shows typical mistakes too.
        idx = torch.randperm(len(targets), generator=torch.Generator().manual_seed(7))[:24]
        images = torch.stack([loader.dataset[i][0] for i in idx.tolist()])
        plot_predictions(images, targets[idx], logits[idx], args.out / "predictions.png")
        print(f"wrote {args.out / 'predictions.png'}")


if __name__ == "__main__":
    main()
