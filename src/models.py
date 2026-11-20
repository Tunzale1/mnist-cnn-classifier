# src/models.py
import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """
    Two-block Convolutional Neural Network trained from scratch for 28x28 grayscale digits.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Block 1: (B, 1, 28, 28) -> (B, 32, 14, 14)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2: (B, 32, 14, 14) -> (B, 64, 7, 7)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Dense Head: (B, 64 * 7 * 7) -> (B, 10)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)


class DigitMLP(nn.Module):
    """
    Plain Multi-Layer Perceptron baseline trained under identical conditions.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cnn = DigitCNN()
    mlp = DigitMLP()
    print(f"DigitCNN trainable params: {count_parameters(cnn):,}")
    print(f"DigitMLP trainable params: {count_parameters(mlp):,}")


class RegularizedDigitCNN(nn.Module):
    """
    CNN enhanced with Batch Normalization and Dropout to reduce overfitting
    and improve generalization on shifted/rotated handwriting.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)    