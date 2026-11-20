import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import DigitCNN, RegularizedDigitCNN
from predict_custom import preprocess_handwritten_image


def load_trained_cnn(device):
    if os.path.exists("models/cnn_mnist_bonus.pth"):
        model = RegularizedDigitCNN().to(device)
        path = "models/cnn_mnist_bonus.pth"
    else:
        model = DigitCNN().to(device)
        path = "models/cnn_mnist.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Loaded filters from {path}")
    return model


def plot_conv1_filters(model, out_path: str):
    # Shape: (32, 1, 3, 3)
    weights = model.conv_block1[0].weight.detach().cpu().numpy()
    vmax = np.abs(weights).max()

    fig, axes = plt.subplots(4, 8, figsize=(11, 5.8))
    for i, ax in enumerate(axes.flat):
        kernel = weights[i, 0]
        ax.imshow(kernel, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"F{i:02d}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Learned Conv1 3x3 Filters (Red = Positive / Blue = Negative Weight)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved filter grid to {out_path}")


def plot_feature_maps(model, image_path: str, device, out_path: str):
    _, _, tensor = preprocess_handwritten_image(image_path)
    with torch.no_grad():
        # Pass through Conv2d + (BatchNorm) + ReLU before MaxPool
        x = tensor.to(device)
        for layer in list(model.conv_block1.children())[:-1]:
            x = layer(x)
        fmaps = x.squeeze(0).cpu().numpy()  # (32, 28, 28)

    fig, axes = plt.subplots(4, 8, figsize=(11, 5.8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fmaps[i], cmap="viridis")
        ax.set_title(f"Map {i:02d}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Conv1 Activation Maps on {os.path.basename(image_path)}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved activation maps to {out_path}")


def main():
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_cnn(device)

    plot_conv1_filters(model, "outputs/conv1_filters.png")

    sample_imgs = sorted(glob.glob("custom_digits/*.png") + glob.glob("custom_digits/*.jpg"))
    if sample_imgs:
        plot_feature_maps(model, sample_imgs[0], device, "outputs/conv1_feature_maps.png")


if __name__ == "__main__":
    main()