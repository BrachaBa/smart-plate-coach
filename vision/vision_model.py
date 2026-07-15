"""EfficientNet-B0 architecture: pretrained backbone + a head sized for our classes.

Spec 2.1: load ImageNet Pre-trained Weights, replace the Fully-Connected head
to match the project's class count, and support the two-phase Transfer
Learning -> Fine-Tuning training strategy (freeze backbone first, unfreeze
later at a lower learning rate).
"""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B0 with ImageNet weights and a fresh (num_classes)-way head."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_inference_transforms():
    """Preprocessing that matches what EfficientNet-B0 was pretrained with.

    Use this (no randomness) for validation/test. Task 8 will define a
    separate train-only transform that adds augmentation on top of this.
    """
    return EfficientNet_B0_Weights.DEFAULT.transforms()


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze every layer except the classifier head.

    Phase 1 (Transfer Learning): trainable=False -> only the new head learns.
    Phase 2 (Fine-Tuning, task 9): trainable=True -> unfreeze upper layers too,
    typically paired with a lower learning rate.
    """
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = trainable
