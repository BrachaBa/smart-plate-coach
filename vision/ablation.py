"""Ablation Study (spec 2.4): run configs 2-4 and compare against the
Baseline (config 1, produced by `vision.train.run_baseline`).

Config definitions (spec 2.4 table):
  1) Baseline       - raw uncleaned data, frozen backbone, no augmentation
  2) + Cleaning      - cleaned data, frozen backbone, no augmentation
  3) + Fine-Tuning   - cleaned data, Transfer Learning -> Fine-Tuning, no augmentation
  4) + Augmentation  - cleaned data, Transfer Learning -> Fine-Tuning, train-only augmentation (full pipeline)

Run `python -m vision.train` first (config 1), then `python -m vision.data_cleaning`
(writes vision/results/train_clean.csv / test_clean.csv), then this module.
"""

import json
from pathlib import Path

import pandas as pd
import torch

from vision.dataset import make_class_to_idx, split_train_val, load_food101_splits
from vision.train import (
    CHECKPOINT_DIR,
    DATA_DIR,
    RESULTS_DIR,
    evaluate_on_test,
    train_model,
    train_transfer_then_finetune,
)
from vision.vision_model import get_inference_transforms, get_train_transforms

CLEAN_TRAIN_CSV = RESULTS_DIR / "train_clean.csv"
CLEAN_TEST_CSV = RESULTS_DIR / "test_clean.csv"


def _load_cleaned_splits():
    """Load the cleaned CSVs written by `vision.data_cleaning.run_cleaning`.

    Cleaning only ever removes rows, so classes.txt (90 classes) is still the
    correct label space — class_to_idx doesn't need to be rebuilt from the
    cleaned data.
    """
    if not CLEAN_TRAIN_CSV.exists() or not CLEAN_TEST_CSV.exists():
        raise FileNotFoundError(
            "Cleaned splits not found. Run `python -m vision.data_cleaning` first "
            f"to produce {CLEAN_TRAIN_CSV} and {CLEAN_TEST_CSV}."
        )
    classes, _, _ = load_food101_splits(DATA_DIR)
    train_df = pd.read_csv(CLEAN_TRAIN_CSV)
    test_df = pd.read_csv(CLEAN_TEST_CSV)
    return classes, train_df, test_df


def _save_config(name: str, model, history: pd.DataFrame, test_metrics: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{name}.pt")
    history.to_csv(RESULTS_DIR / f"{name}_history.csv", index=False)
    with open(RESULTS_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


def run_config_2_cleaning(epochs=5, batch_size=64, lr=1e-3, num_workers=0):
    """+ Cleaning: cleaned data, still frozen backbone, no augmentation."""
    classes, train_df, test_df = _load_cleaned_splits()
    class_to_idx = make_class_to_idx(classes)
    train_split, val_split = split_train_val(train_df, val_fraction=0.1)

    eval_transform = get_inference_transforms()
    train_transform = eval_transform  # no augmentation yet in this config

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
        run_name="config2_cleaning",
    )

    test_metrics = evaluate_on_test(model, test_df, class_to_idx, eval_transform, batch_size=batch_size, num_workers=num_workers)
    print("[config2_cleaning] test metrics:", test_metrics)
    _save_config("config2_cleaning", model, history, test_metrics)
    return model, history, test_metrics


def run_config_3_finetuning(
    transfer_epochs=3, finetune_epochs=3, batch_size=64, transfer_lr=1e-3, finetune_lr=1e-4, num_workers=0
):
    """+ Fine-Tuning: cleaned data, Transfer Learning then Fine-Tuning, no augmentation."""
    classes, train_df, test_df = _load_cleaned_splits()
    class_to_idx = make_class_to_idx(classes)
    train_split, val_split = split_train_val(train_df, val_fraction=0.1)

    eval_transform = get_inference_transforms()
    train_transform = eval_transform  # no augmentation yet in this config

    model, history = train_transfer_then_finetune(
        train_split,
        val_split,
        class_to_idx,
        len(classes),
        train_transform,
        eval_transform,
        transfer_epochs=transfer_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
        transfer_lr=transfer_lr,
        finetune_lr=finetune_lr,
        num_workers=num_workers,
        run_name="config3_finetune",
    )

    test_metrics = evaluate_on_test(model, test_df, class_to_idx, eval_transform, batch_size=batch_size, num_workers=num_workers)
    print("[config3_finetune] test metrics:", test_metrics)
    _save_config("config3_finetune", model, history, test_metrics)
    return model, history, test_metrics


def run_config_4_full(
    transfer_epochs=3, finetune_epochs=3, batch_size=64, transfer_lr=1e-3, finetune_lr=1e-4, num_workers=0
):
    """+ Augmentation: cleaned data, Transfer Learning then Fine-Tuning, train-only augmentation. Full pipeline."""
    classes, train_df, test_df = _load_cleaned_splits()
    class_to_idx = make_class_to_idx(classes)
    train_split, val_split = split_train_val(train_df, val_fraction=0.1)

    eval_transform = get_inference_transforms()
    train_transform = get_train_transforms()  # augmentation applied to Train only

    model, history = train_transfer_then_finetune(
        train_split,
        val_split,
        class_to_idx,
        len(classes),
        train_transform,
        eval_transform,
        transfer_epochs=transfer_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
        transfer_lr=transfer_lr,
        finetune_lr=finetune_lr,
        num_workers=num_workers,
        run_name="config4_full",
    )

    test_metrics = evaluate_on_test(model, test_df, class_to_idx, eval_transform, batch_size=batch_size, num_workers=num_workers)
    print("[config4_full] test metrics:", test_metrics)
    _save_config("config4_full", model, history, test_metrics)
    return model, history, test_metrics


def summarize_ablation():
    """Collect all 4 configs' test metrics into one table and check the
    spec 2.5 success criterion (config 4 >= 15% better than config 1).
    """
    names = ["baseline", "config2_cleaning", "config3_finetune", "config4_full"]
    rows = []
    for name in names:
        metrics_path = RESULTS_DIR / f"{name}_metrics.json"
        if not metrics_path.exists():
            print(f"[summary] missing {metrics_path}, skipping")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append({"config": name, **metrics})

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)
    print(summary)

    if {"baseline", "config4_full"}.issubset(summary["config"].values):
        baseline_acc = summary.loc[summary["config"] == "baseline", "test_accuracy"].iloc[0]
        final_acc = summary.loc[summary["config"] == "config4_full", "test_accuracy"].iloc[0]
        improvement = (final_acc - baseline_acc) / baseline_acc * 100
        print(f"[summary] accuracy improvement over baseline: {improvement:.1f}% (target: >= 15%)")

    return summary


if __name__ == "__main__":
    run_config_2_cleaning()
    run_config_3_finetuning()
    run_config_4_full()
    summarize_ablation()
