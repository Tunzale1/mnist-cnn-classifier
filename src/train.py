# src/train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data import get_dataloaders
from models import DigitCNN, DigitMLP


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.concatenate(all_preds), np.concatenate(all_targets)


def train_model(model, name, train_loader, val_loader, epochs, lr, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n--- Training {name} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%"
        )

    return history


def plot_comparison_curves(cnn_hist, mlp_hist, out_path):
    epochs = range(1, len(cnn_hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curves
    axes[0].plot(epochs, cnn_hist["train_loss"], "b-o", label="CNN Train Loss")
    axes[0].plot(epochs, cnn_hist["val_loss"], "b--s", label="CNN Val Loss")
    axes[0].plot(epochs, mlp_hist["train_loss"], "r-o", label="MLP Train Loss")
    axes[0].plot(epochs, mlp_hist["val_loss"], "r--s", label="MLP Val Loss")
    axes[0].set_title("Cross-Entropy Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy Curves
    axes[1].plot(epochs, [a * 100 for a in cnn_hist["train_acc"]], "b-o", label="CNN Train Acc")
    axes[1].plot(epochs, [a * 100 for a in cnn_hist["val_acc"]], "b--s", label="CNN Val Acc")
    axes[1].plot(epochs, [a * 100 for a in mlp_hist["train_acc"]], "r-o", label="MLP Train Acc")
    axes[1].plot(epochs, [a * 100 for a in mlp_hist["val_acc"]], "r--s", label="MLP Val Acc")
    axes[1].set_title("Accuracy (%) per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("CNN Test Set Confusion Matrix (10,000 Samples)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Identify top 3 most-confused pairs (off-diagonal)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat_indices = np.argsort(cm_no_diag.ravel())[::-1][:3]
    print("\nTop Confused Digit Pairs on Test Set:")
    for idx in flat_indices:
        true_digit, pred_digit = divmod(idx, 10)
        count = cm_no_diag[true_digit, pred_digit]
        print(f"  True {true_digit} predicted as {pred_digit}: {count} errors")


def main():
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 10
    lr = 1e-3
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64, seed=42)

    mlp = DigitMLP().to(device)
    cnn = DigitCNN().to(device)

    mlp_path = "models/mlp_mnist.pth"
    cnn_path = "models/cnn_mnist.pth"

    mlp_hist = train_model(mlp, "MLP Baseline", train_loader, val_loader, epochs, lr, device, mlp_path)
    cnn_hist = train_model(cnn, "DigitCNN", train_loader, val_loader, epochs, lr, device, cnn_path)

    plot_comparison_curves(cnn_hist, mlp_hist, "outputs/training_curves.png")

    # Reload best validation checkpoints for final test comparison
    criterion = nn.CrossEntropyLoss()
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))

    mlp_test_loss, mlp_test_acc, _, _ = evaluate(mlp, test_loader, criterion, device)
    cnn_test_loss, cnn_test_acc, cnn_preds, cnn_targets = evaluate(cnn, test_loader, criterion, device)

    print("\n=== Final Test Set Results ===")
    print(f"MLP Baseline Test Accuracy: {mlp_test_acc*100:.2f}% (Loss: {mlp_test_loss:.4f})")
    print(f"DigitCNN     Test Accuracy: {cnn_test_acc*100:.2f}% (Loss: {cnn_test_loss:.4f})")

    plot_confusion_matrix(cnn_targets, cnn_preds, "outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()