# src/predict_custom.py
import os
import glob
import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from models import DigitCNN
from data import MNIST_MEAN, MNIST_STD


def preprocess_handwritten_image(image_path: str):
    """
    Explicitly converts a real-world handwritten digit photo/drawing into an MNIST-compatible tensor:
      1. Grayscale conversion
      2. Polarity check & inversion (MNIST expects white ink on black background)
      3. Bounding-box crop & aspect-preserving resize to 20x20
      4. Center-of-mass padding into 28x28
      5. Scaling to [0, 1] and MNIST standardization
    """
    raw_img = Image.open(image_path).convert("L")
    arr = np.array(raw_img, dtype=np.float32)

    # 1. Auto-invert if background is light (paper/white canvas)
    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border_pixels.mean() > 127.0:
        arr = 255.0 - arr

    # 2. Denoise faint paper shadows / compression artifacts
    arr[arr < 40.0] = 0.0

    # 3. Crop tight bounding box around the digit
    ys, xs = np.where(arr > 0)
    if len(ys) > 0 and len(xs) > 0:
        cropped = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        cropped = arr

    # 4. Resize longest edge to 20 pixels (matching LeCun's MNIST pipeline)
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = np.array(
        Image.fromarray(cropped.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    # 5. Place into 28x28 canvas and shift by center-of-mass
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    cy, cx = np.indices(canvas.shape)
    total_mass = canvas.sum()
    if total_mass > 0:
        shift_y = int(np.clip(round(14.0 - (cy * canvas).sum() / total_mass), -4, 4))
        shift_x = int(np.clip(round(14.0 - (cx * canvas).sum() / total_mass), -4, 4))
        canvas = np.roll(canvas, shift=(shift_y, shift_x), axis=(0, 1))

    # 6. Normalize to match training distribution
    scaled = canvas / 255.0
    normalized = (scaled - MNIST_MEAN[0]) / MNIST_STD[0]
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)

    return raw_img, canvas, tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "models/cnn_mnist.pth"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Saved weights not found at {weights_path}. Run src/train.py first.")

    # Reload model from disk without retraining
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Successfully reloaded weights from {weights_path}")

    image_paths = sorted(
        glob.glob("custom_digits/*.png")
        + glob.glob("custom_digits/*.jpg")
        + glob.glob("custom_digits/*.jpeg")
    )

    if not image_paths:
        print("Place at least 5 handwritten digit images inside ./custom_digits/ and re-run.")
        return

    fig, axes = plt.subplots(len(image_paths), 2, figsize=(6, 2.6 * len(image_paths)))
    if len(image_paths) == 1:
        axes = np.expand_dims(axes, axis=0)

    print("\n--- Custom Handwritten Digit Predictions ---")
    with torch.no_grad():
        for idx, path in enumerate(image_paths):
            raw_img, processed_28x28, tensor = preprocess_handwritten_image(path)
            logits = model(tensor.to(device))
            probs = torch.softmax(logits, dim=1)[0]
            pred_digit = probs.argmax().item()
            confidence = probs[pred_digit].item() * 100

            print(f"{os.path.basename(path):20s} -> Predicted: {pred_digit} ({confidence:.1f}% confidence)")

            axes[idx, 0].imshow(raw_img, cmap="gray")
            axes[idx, 0].set_title(f"Raw: {os.path.basename(path)}")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(processed_28x28, cmap="gray")
            axes[idx, 1].set_title(f"28x28 Input -> Pred: {pred_digit} ({confidence:.1f}%)")
            axes[idx, 1].axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/custom_predictions.png", dpi=150)
    plt.close()
    print("\nSaved visual verification grid to outputs/custom_predictions.png")


if __name__ == "__main__":
    main()