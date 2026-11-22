"""One-time export of the trained vision checkpoint to ONNX.

The agent runs the classifier with onnxruntime instead of PyTorch, so the CLI
(and the PyInstaller agent.exe) do not need torch at runtime: torch alone would
make the exe 1-2 GB, while onnxruntime keeps it small.

Run once after training (needs torch + torchvision, i.e. the vision/ extras):

    python -m scripts.export_onnx

Reads  vision/checkpoints/config4_full.pt
Writes agent/models/food_classifier.onnx   (committed, so a fresh clone runs)
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from agent.class_names import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "vision" / "checkpoints" / "config4_full.pt"
OUTPUT = ROOT / "agent" / "models" / "food_classifier.onnx"


def main():
    # Same architecture as vision.vision_model.build_model, but without
    # downloading ImageNet weights: the checkpoint overwrites them anyway.
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        OUTPUT,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {CHECKPOINT.name} -> {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
