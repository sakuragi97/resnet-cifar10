"""Training and evaluation loops."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> tuple[float, float]:
    """Run one pass over ``loader`` and return the mean loss and accuracy."""
    model.train()
    # Accumulate on the device so the loop does not sync with the GPU on every step.
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    seen = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum()
        seen += targets.size(0)

    return total_loss.item() / seen, correct.item() / seen


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return the logits and targets for the whole loader, on the CPU."""
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        all_logits.append(model(images.to(device, non_blocking=True)).float().cpu())
        all_targets.append(targets)
    return torch.cat(all_logits), torch.cat(all_targets)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return the mean loss and accuracy on ``loader``."""
    logits, targets = predict(model, loader, device)
    loss = F.cross_entropy(logits, targets).item()
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, accuracy
