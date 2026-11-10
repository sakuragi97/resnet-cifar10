"""Train a ResNet on CIFAR-10.

    python -m resnet.train --config configs/resnet20.yaml
    python -m resnet.train --config configs/resnet56.yaml --epochs 100 --lr 0.05

Writes config.yaml, history.csv (one row per epoch) and model.pt to <out_dir>/<name>/.
"""

import csv
import time
from dataclasses import asdict
from pathlib import Path

import torch

from resnet.config import Config, parse_config, save_config
from resnet.data import build_dataset, build_loader
from resnet.engine import evaluate, train_one_epoch
from resnet.model import ResNet
from resnet.utils import count_parameters, resolve_device, save_checkpoint, seed_everything

HISTORY_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "seconds"]


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config):
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)
    raise ValueError(f"unknown scheduler {config.scheduler!r}, expected 'cosine' or 'multistep'")


def main(argv: list[str] | None = None) -> None:
    config = parse_config(argv)
    seed_everything(config.seed)
    device = resolve_device(config.device)

    run_dir = Path(config.out_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    workers = config.num_workers
    train_set = build_dataset(config.data_dir, train=True)
    test_set = build_dataset(config.data_dir, train=False)
    train_loader = build_loader(train_set, config.batch_size, train=True, num_workers=workers)
    test_loader = build_loader(test_set, 2 * config.batch_size, train=False, num_workers=workers)

    model = ResNet(config.depth, residual=config.residual, zero_init_residual=config.zero_init_residual)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(optimizer, config)
    print(f"{config.name}: {count_parameters(model) / 1e6:.2f}M parameters, training on {device}")

    with open(run_dir / "history.csv", "w", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, device)
            scheduler.step()
            seconds = time.perf_counter() - start

            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "seconds": round(seconds, 1),
                }
            )
            history_file.flush()  # keep the log usable if training is interrupted
            print(
                f"epoch {epoch:3d}/{config.epochs} | lr {lr:.4f} | "
                f"train loss {train_loss:.3f} acc {train_acc:.2%} | "
                f"test loss {test_loss:.3f} acc {test_acc:.2%} | {seconds:.0f}s"
            )

    save_checkpoint(run_dir / "model.pt", model, asdict(config))
    print(f"saved {run_dir / 'model.pt'} (final test accuracy {test_acc:.2%})")


if __name__ == "__main__":
    main()
