"""Seeding, device selection and checkpoint I/O."""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from resnet.model import ResNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: Path, model: ResNet, config: dict) -> None:
    torch.save({"model": model.state_dict(), "config": config}, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[ResNet, dict]:
    """Rebuild a model from a checkpoint written by ``save_checkpoint`` or from a bare state dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    config = checkpoint.get("config", {})

    # A bare state dict has no config: recover the depth from the number of blocks per stage.
    blocks_per_stage = len({key.split(".")[1] for key in state if key.startswith("stage1.")})
    depth = config.get("depth", 6 * blocks_per_stage + 2)

    model = ResNet(depth=depth, residual=config.get("residual", True))
    model.load_state_dict(state)
    return model.to(device).eval(), config
