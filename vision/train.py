"""Training loop shared by every Ablation Study config (spec 2.4).

Running this file directly trains Ablation config (1): the Baseline — raw,
uncleaned data, frozen backbone, no augmentation. Tasks 9 and 11 reuse
`train_model`/`evaluate_on_test` with different data/transform/freeze
arguments instead of duplicating this loop for every config.
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from vision.dataset import (
    FoodImageDataset,
    load_food101_splits,
    make_class_to_idx,
    split_train_val,
)
from vision.vision_model import build_model, get_inference_transforms, set_backbone_trainable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "food41"
CHECKPOINT_DIR = PROJECT_ROOT / "vision" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "vision" / "results"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, all_preds, all_labels = 0.0, [], []
    grad_context = torch.enable_grad() if is_train else torch.no_grad()

    with grad_context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def train_model(
    train_df,
    val_df,
    class_to_idx,
    num_classes,
    train_transform,
    eval_transform,
    freeze_backbone,
    epochs,
    batch_size=64,
    lr=1e-3,
    num_workers=0,
    run_name="run",
):
    """Train, validate each epoch, and return (model, history_df).

    freeze_backbone=True is the Transfer Learning phase (spec 2.1): only the
    new classifier head trains. False is Fine-Tuning: upper layers unfreeze
    too, normally paired with a lower lr by the caller.
    """
    device = get_device()
    print(
        f"[{run_name}] device={device} train={len(train_df)} val={len(val_df)} "
        f"epochs={epochs} freeze_backbone={freeze_backbone}"
    )

    train_ds = FoodImageDataset(train_df, class_to_idx, transform=train_transform)
    val_ds = FoodImageDataset(val_df, class_to_idx, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes).to(device)
    set_backbone_trainable(model, trainable=not freeze_backbone)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        elapsed = time.time() - start

        print(
            f"[{run_name}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    return model, pd.DataFrame(history)


def evaluate_on_test(model, test_df, class_to_idx, eval_transform, batch_size=64, num_workers=0):
    """Final Accuracy + macro-F1 on the held-out test set (spec 2.5, ablation bars)."""
    device = get_device()
    test_ds = FoodImageDataset(test_df, class_to_idx, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    return {
        "test_accuracy": accuracy_score(all_labels, all_preds),
        "test_f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }


def run_baseline(epochs=5, batch_size=64, lr=1e-3, num_workers=0):
    """Ablation config (1): raw uncleaned data, frozen backbone, no augmentation."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    classes, train_df, test_df = load_food101_splits(DATA_DIR)
    class_to_idx = make_class_to_idx(classes)
    train_split, val_split = split_train_val(train_df, val_fraction=0.1)

    eval_transform = get_inference_transforms()
    # No augmentation for the baseline: train images get the same plain
    # preprocessing as val/test. The baseline must exclude every later
    # intervention (cleaning, augmentation, fine-tuning) so the 15%
    # improvement claim checked in task 12 is measured against a real floor.
    train_transform = eval_transform

    model, history = train_model(
        train_split,
        val_split,
        class_to_idx,
        len(classes),
        train_transform,
        eval_transform,
        freeze_backbone=True,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        run_name="baseline",
    )

    test_metrics = evaluate_on_test(model, test_df, class_to_idx, eval_transform, batch_size=batch_size, num_workers=num_workers)
    print("[baseline] test metrics:", test_metrics)

    torch.save(model.state_dict(), CHECKPOINT_DIR / "baseline.pt")
    history.to_csv(RESULTS_DIR / "baseline_history.csv", index=False)
    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, history, test_metrics


if __name__ == "__main__":
    run_baseline()
