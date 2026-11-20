# src/data.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def get_dataloaders(data_dir: str = "./data", batch_size: int = 64, seed: int = 42):
    transform = get_transforms()

    full_train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Explicit 50k train / 10k validation split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [50_000, 10_000], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_l, val_l, test_l = get_dataloaders()
    x_batch, y_batch = next(iter(train_l))
    print(f"Train samples: {len(train_l.dataset)} | Val: {len(val_l.dataset)} | Test: {len(test_l.dataset)}")
    print(f"Batch shape: {x_batch.shape} | Normalized mean: {x_batch.mean():.4f}, std: {x_batch.std():.4f}")