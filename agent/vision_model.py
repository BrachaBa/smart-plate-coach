"""Inference wrapper around the Part א' (Deep Learning) trained classifier.

The trained PyTorch checkpoint (vision/checkpoints/config4_full.pt) is exported
once to ONNX by scripts/export_onnx.py, and the agent runs that file with
onnxruntime. This keeps torch out of the agent's runtime: a fresh clone runs
without training anything, and the PyInstaller agent.exe stays small.

Preprocessing reproduces EfficientNet_B0_Weights.DEFAULT.transforms() exactly
(resize shorter side to 256 bicubic, center-crop 224, scale to [0,1],
ImageNet mean/std normalization), so predictions match the PyTorch model.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from agent.class_names import CLASS_NAMES


def _resource_root() -> Path:
    """Project root when run from source; PyInstaller's unpack dir inside agent.exe."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


MODEL_PATH = _resource_root() / "agent" / "models" / "food_classifier.onnx"

RESIZE_SIZE = 256
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None  # onnxruntime.InferenceSession, loaded lazily once per process


def load_model():
    """Load the ONNX classifier once per process and cache it."""
    global _session
    if _session is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Vision model not found at {MODEL_PATH}. "
                "Run `python -m scripts.export_onnx` after training (see TRAINING.md)."
            )
        import onnxruntime as ort

        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL image -> (1, 3, 224, 224) float32, matching the torchvision transform."""
    width, height = image.size
    scale = RESIZE_SIZE / min(width, height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if width <= height:
        new_size = (RESIZE_SIZE, new_size[1])
    else:
        new_size = (new_size[0], RESIZE_SIZE)
    image = image.resize(new_size, Image.BICUBIC)

    width, height = image.size
    left = int(round((width - CROP_SIZE) / 2.0))
    top = int(round((height - CROP_SIZE) / 2.0))
    image = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return array.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def classify_image(image_path: str, top_k: int = 3) -> dict:
    """Classify a meal photo into the top-k Food-101 classes.

    Returns {"foods": [{"name": str, "confidence": float}, ...], "portion_size": str}.

    Portion size is not something Part א' was trained to estimate -- the checkpoint
    is a pure classifier with no size/regression head. This is a known simplification:
    every image is reported as "medium" rather than inventing an unvalidated heuristic
    that would look more rigorous than it is.

    Raises FileNotFoundError if the path doesn't exist, or ValueError if the file
    isn't a readable image -- callers (the CLI, the history node) catch these and
    tell the user, instead of crashing.
    """
    path = Path(image_path.strip().strip('"').strip("'"))
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        # Phone photos store their orientation in EXIF; without applying it a
        # portrait shot reaches the classifier rotated by 90 degrees.
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unsupported or corrupted image file: {image_path}") from exc

    session = load_model()
    logits = session.run(None, {"image": preprocess(image)})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    k = min(top_k, len(CLASS_NAMES))
    top_idx = np.argsort(probs)[::-1][:k]
    foods = [{"name": CLASS_NAMES[i], "confidence": float(probs[i])} for i in top_idx]

    return {"foods": foods, "portion_size": "medium"}
