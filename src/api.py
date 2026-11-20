import io
import os
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

from models import DigitCNN, RegularizedDigitCNN
from data import MNIST_MEAN, MNIST_STD

app = FastAPI(title="MNIST Handwritten Digit Classifier API", version="1.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists("models/cnn_mnist_bonus.pth"):
    model = RegularizedDigitCNN().to(device)
    WEIGHTS_PATH = "models/cnn_mnist_bonus.pth"
else:
    model = DigitCNN().to(device)
    WEIGHTS_PATH = "models/cnn_mnist.pth"

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()


def preprocess_bytes(image_bytes: bytes) -> torch.Tensor:
    raw_img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(raw_img, dtype=np.float32)

    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border_pixels.mean() > 127.0:
        arr = 255.0 - arr

    arr[arr < 40.0] = 0.0
    ys, xs = np.where(arr > 0)
    cropped = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) > 0 else arr

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = np.array(
        Image.fromarray(cropped.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    cy, cx = np.indices(canvas.shape)
    total_mass = canvas.sum()
    if total_mass > 0:
        shift_y = int(np.clip(round(14.0 - (cy * canvas).sum() / total_mass), -4, 4))
        shift_x = int(np.clip(round(14.0 - (cx * canvas).sum() / total_mass), -4, 4))
        canvas = np.roll(canvas, shift=(shift_y, shift_x), axis=(0, 1))

    scaled = canvas / 255.0
    normalized = (scaled - MNIST_MEAN[0]) / MNIST_STD[0]
    return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


@app.get("/health")
def health():
    return {"status": "ok", "weights_loaded": WEIGHTS_PATH, "device": str(device)}


@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    tensor = preprocess_bytes(image_bytes).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    pred_digit = int(probs.argmax())
    return {
        "filename": file.filename,
        "predicted_digit": pred_digit,
        "confidence": round(float(probs[pred_digit]), 4),
        "probabilities": {str(i): round(float(probs[i]), 4) for i in range(10)},
    }