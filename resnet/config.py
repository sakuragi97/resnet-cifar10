"""Training configuration: dataclass defaults, overridden by a YAML file, overridden by CLI flags."""

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    name: str = "resnet20"
    # model
    depth: int = 20
    residual: bool = True
    zero_init_residual: bool = False
    # optimization
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"  # "cosine" or "multistep"
    milestones: list[int] = field(default_factory=list)  # epochs at which "multistep" divides lr by 10
    # runtime
    data_dir: str = "data"
    out_dir: str = "runs"
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"


def parse_config(argv: list[str] | None = None) -> Config:
    """Build a Config from ``--config file.yaml`` plus any ``--<field> value`` overrides."""
    parser = argparse.ArgumentParser(description="Train a ResNet on CIFAR-10.")
    parser.add_argument("--config", type=Path, help="YAML file with Config fields")
    for f in fields(Config):
        if f.type is bool:
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction)
        elif f.name == "milestones":
            parser.add_argument(f"--{f.name}", type=int, nargs="+")
        else:
            parser.add_argument(f"--{f.name}", type=f.type)
    args = vars(parser.parse_args(argv))

    config_path = args.pop("config")
    values = {}
    if config_path:
        values = yaml.safe_load(config_path.read_text()) or {}
    values.update({key: value for key, value in args.items() if value is not None})
    return Config(**values)  # unknown keys in the YAML file raise a TypeError


def save_config(config: Config, path: Path) -> None:
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
