"""Inference wrapper around the Part א' (Deep Learning) trained classifier.

Reuses vision/vision_model.py (architecture + transforms) and vision/train.py
(device detection) as-is -- no retraining or duplicated model-building logic here.
"""

from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from vision.train import get_device
from vision.vision_model import build_model, get_inference_transforms

from agent.class_names import CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "vision" / "checkpoints" / "config4_full.pt"

_model_singleton = None  # (model, device), loaded lazily once per process


def load_model():
    """Load the trained checkpoint once per process and cache it."""
    global _model_singleton
    if _model_singleton is None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Vision checkpoint not found at {CHECKPOINT_PATH}. "
                "Run the Part א' training pipeline first (see TRAINING.md)."
            )
        device = get_device()
        model = build_model(len(CLASS_NAMES))
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        _model_singleton = (model, device)
    return _model_singleton


def classify_image(image_path: str, top_k: int = 3) -> dict:
    """Classify a meal photo into the top-k Food-101 classes.

    Returns {"foods": [{"name": str, "confidence": float}, ...], "portion_size": str}.

    Portion size is not something Part א' was trained to estimate -- the checkpoint
    is a pure classifier with no size/regression head. This is a known simplification:
    every image is reported as "medium" rather than inventing an unvalidated heuristic
    that would look more rigorous than it is.

    Raises FileNotFoundError if the path doesn't exist, or ValueError if the file
    isn't a readable image -- callers (the input gate node, the CLI) are responsible
    for catching these and prompting the user again, not crashing.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupted image file: {image_path}") from exc

    model, device = load_model()
    transform = get_inference_transforms()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    k = min(top_k, len(CLASS_NAMES))
    top_probs, top_idx = torch.topk(probs, k)
    foods = [
        {"name": CLASS_NAMES[idx], "confidence": float(prob)}
        for prob, idx in zip(top_probs.tolist(), top_idx.tolist())
    ]

    return {"foods": foods, "portion_size": "medium"}
