"""Evaluate a checkpoint on the CIFAR-10 test set.

    python -m resnet.evaluate runs/resnet20/model.pt

Prints overall and per-class accuracy and writes metrics.json next to the checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from resnet.data import CLASSES, build_dataset, build_loader
from resnet.engine import predict
from resnet.utils import count_parameters, load_checkpoint, resolve_device


def confusion_matrix(preds: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """Rows are true classes, columns are predicted classes."""
    flat = targets * num_classes + preds
    return torch.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ResNet checkpoint on CIFAR-10.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    loader = build_loader(
        build_dataset(args.data_dir, train=False), args.batch_size, train=False, num_workers=args.num_workers
    )

    logits, targets = predict(model, loader, device)
    cm = confusion_matrix(logits.argmax(dim=1), targets, len(CLASSES))
    per_class = (cm.diag() / cm.sum(dim=1)).tolist()
    accuracy = (cm.diag().sum() / cm.sum()).item()

    print(f"ResNet-{model.depth} ({count_parameters(model):,} parameters)")
    print(f"test accuracy {accuracy:.2%}  |  test error {1 - accuracy:.2%}\n")
    for name, acc in sorted(zip(CLASSES, per_class, strict=True), key=lambda item: -item[1]):
        print(f"  {name:<11} {acc:6.2%}  {'#' * round(acc * 40)}")

    metrics = {
        "depth": model.depth,
        "parameters": count_parameters(model),
        "accuracy": accuracy,
        "per_class_accuracy": dict(zip(CLASSES, per_class, strict=True)),
        "confusion_matrix": cm.tolist(),
    }
    out = args.checkpoint.with_name("metrics.json")
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
