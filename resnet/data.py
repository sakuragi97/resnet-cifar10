"""CIFAR-10 datasets, loaders and the augmentation from Section 4.2 of the paper."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# Per-channel mean and std of the 50k training images.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def build_transform(train: bool) -> transforms.Compose:
    """Training: pad 4 px, random 32x32 crop, random horizontal flip. Test: normalize only."""
    augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose([*augment, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build_dataset(root: str, train: bool) -> datasets.CIFAR10:
    return datasets.CIFAR10(root, train=train, transform=build_transform(train), download=True)


def build_loader(dataset, batch_size: int, train: bool, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        # Keep the training workers alive between epochs; test workers only live during evaluation.
        persistent_workers=train and num_workers > 0,
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Map normalized images back to [0, 1] for plotting."""
    mean = torch.tensor(MEAN, device=images.device).view(-1, 1, 1)
    std = torch.tensor(STD, device=images.device).view(-1, 1, 1)
    return (images * std + mean).clamp(0, 1)
