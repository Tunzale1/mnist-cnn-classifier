import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from data import MNIST_MEAN, MNIST_STD
from models import DigitCNN, RegularizedDigitCNN
from train import set_seed, train_model, evaluate


class TransformSubset(Dataset):
    """Applies separate transforms to train vs. validation subsets without leakage."""
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        return self.transform(img), label


def get_ablation_loaders(data_dir: str = "./data", batch_size: int = 64, seed: int = 42):
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    aug_transform = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    raw_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(range(len(raw_train)), [50_000, 10_000], generator=generator)

    clean_train_ds = TransformSubset(raw_train, train_sub.indices, eval_transform)
    aug_train_ds = TransformSubset(raw_train, train_sub.indices, aug_transform)
    val_ds = TransformSubset(raw_train, val_sub.indices, eval_transform)

    clean_train_loader = DataLoader(clean_train_ds, batch_size=batch_size, shuffle=True)
    aug_train_loader = DataLoader(aug_train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return clean_train_loader, aug_train_loader, val_loader, test_loader


def plot_ablation(histories: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {
        "1. Unregularized CNN": ("#7f8c8d", "--"),
        "2. Regularized (BN + Dropout)": ("#2980b9", "-."),
        "3. Regularized + Augmented": ("#27ae60", "-"),
    }

    for label, hist in histories.items():
        epochs = range(1, len(hist["val_loss"]) + 1)
        color, ls = styles[label]
        axes[0].plot(epochs, hist["val_loss"], color=color, linestyle=ls, marker="o", markersize=4, label=label)
        axes[1].plot(epochs, [a * 100 for a in hist["val_acc"]], color=color, linestyle=ls, marker="o", markersize=4, label=label)

    axes[0].set_title("Validation Loss: Regularization & Augmentation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Accuracy (%): Regularization & Augmentation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 12
    lr = 1e-3
    clean_train_loader, aug_train_loader, val_loader, test_loader = get_ablation_loaders(batch_size=64, seed=42)

    experiments = [
        ("1. Unregularized CNN", DigitCNN().to(device), clean_train_loader, "models/cnn_unreg.pth"),
        ("2. Regularized (BN + Dropout)", RegularizedDigitCNN().to(device), clean_train_loader, "models/cnn_reg.pth"),
        ("3. Regularized + Augmented", RegularizedDigitCNN().to(device), aug_train_loader, "models/cnn_mnist_bonus.pth"),
    ]

    histories = {}
    criterion = nn.CrossEntropyLoss()
    results = []

    for name, model, loader, save_path in experiments:
        hist = train_model(model, name, loader, val_loader, epochs, lr, device, save_path)
        histories[name] = hist

        model.load_state_dict(torch.load(save_path, map_location=device))
        _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        results.append((name, max(hist["val_acc"]) * 100, test_acc * 100))

    plot_ablation(histories, "outputs/bonus_ablation_curves.png")

    print("\n=== Bonus Ablation Summary ===")
    print(f"{'Model Variant':32s} | {'Best Val Acc':>12s} | {'Test Acc':>10s}")
    print("-" * 62)
    for name, best_val, test_acc in results:
        print(f"{name:32s} | {best_val:11.2f}% | {test_acc:9.2f}%")


if __name__ == "__main__":
    main()