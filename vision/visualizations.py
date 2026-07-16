"""The 4 required visualizations (spec 2.4): Class Distribution, Learning
Curves, Confusion Matrix, and Ablation Study bar chart.

Run after `vision.train` (baseline), `vision.data_cleaning`, and
`vision.ablation` have all produced their CSV/JSON outputs.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from vision.dataset import FoodImageDataset, load_food101_splits, make_class_to_idx
from vision.train import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR, get_device
from vision.vision_model import build_model, get_inference_transforms

FIGURES_DIR = RESULTS_DIR / "figures"


def plot_class_distribution():
    """Train-set images per class, before vs after cleaning (spec 2.4)."""
    classes, train_df, _ = load_food101_splits(DATA_DIR)
    train_clean = pd.read_csv(RESULTS_DIR / "train_clean.csv")

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    sns.countplot(data=train_df, x="class", order=classes["class"], ax=axes[0])
    axes[0].set_title("Train set: images per class (before cleaning)")
    axes[0].tick_params(axis="x", rotation=90)

    sns.countplot(data=train_clean, x="class", order=classes["class"], ax=axes[1])
    axes[1].set_title("Train set: images per class (after cleaning)")
    axes[1].tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[viz] saved {FIGURES_DIR / 'class_distribution.png'}")


def plot_learning_curves():
    """Loss/Accuracy over epochs, Train vs Validation, including the
    Transfer Learning -> Fine-Tuning transition (spec 2.4). One figure per
    ablation config that has a saved history.
    """
    configs = {
        "baseline": "Baseline",
        "config2_cleaning": "+ Cleaning",
        "config3_finetune": "+ Fine-Tuning",
        "config4_full": "+ Augmentation (full pipeline)",
    }

    for config_name, title in configs.items():
        history_path = RESULTS_DIR / f"{config_name}_history.csv"
        if not history_path.exists():
            print(f"[viz] skipping {config_name}: {history_path} not found")
            continue

        history = pd.read_csv(history_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history["epoch"], history["train_loss"], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
        axes[0].set_title(f"{title}: Loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend()

        axes[1].plot(history["epoch"], history["train_acc"], label="train")
        axes[1].plot(history["epoch"], history["val_acc"], label="val")
        axes[1].set_title(f"{title}: Accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].legend()

        # Mark the Transfer Learning -> Fine-Tuning boundary, if this config has one.
        if "phase" in history.columns and history["phase"].nunique() > 1:
            transfer_epochs = (history["phase"] == "transfer_learning").sum()
            for ax in axes:
                ax.axvline(transfer_epochs + 0.5, color="gray", linestyle="--", label="fine-tuning starts")

        fig.tight_layout()
        out_path = FIGURES_DIR / f"learning_curves_{config_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[viz] saved {out_path}")


def plot_confusion_matrix(config_name="config4_full", batch_size=64, sample_per_class=None):
    """Normalized confusion matrix for the final model on the test set (spec 2.4).

    sample_per_class limits how many test images per class are used, to keep
    this fast — pass None to use the full test set.
    """
    checkpoint_path = CHECKPOINT_DIR / f"{config_name}.pt"
    if not checkpoint_path.exists():
        print(f"[viz] skipping confusion matrix: {checkpoint_path} not found")
        return

    classes, _, test_df = load_food101_splits(DATA_DIR)
    class_to_idx = make_class_to_idx(classes)

    if sample_per_class:
        test_df = test_df.groupby("class").head(sample_per_class).reset_index(drop=True)

    device = get_device()
    model = build_model(len(classes)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    eval_transform = get_inference_transforms()
    test_ds = FoodImageDataset(test_df, class_to_idx, transform=eval_transform)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    class_names = list(class_to_idx.keys())

    fig, ax = plt.subplots(figsize=(24, 22))
    sns.heatmap(cm, annot=False, xticklabels=class_names, yticklabels=class_names, cmap="viridis", ax=ax)
    ax.set_title(f"Normalized confusion matrix ({config_name})")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", rotation=0, labelsize=6)

    fig.tight_layout()
    out_path = FIGURES_DIR / f"confusion_matrix_{config_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] saved {out_path}")


def plot_ablation_bars():
    """Accuracy/F1 across the 4 ablation configs (spec 2.4)."""
    summary_path = RESULTS_DIR / "ablation_summary.csv"
    if not summary_path.exists():
        print(f"[viz] skipping ablation bars: {summary_path} not found")
        return

    summary = pd.read_csv(summary_path)
    order = ["baseline", "config2_cleaning", "config3_finetune", "config4_full"]
    summary["config"] = pd.Categorical(summary["config"], categories=order, ordered=True)
    summary = summary.sort_values("config")

    melted = summary.melt(id_vars="config", value_vars=["test_accuracy", "test_f1_macro"], var_name="metric")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="config", y="value", hue="metric", ax=ax)
    ax.set_title("Ablation Study: Accuracy / F1 across configs")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out_path = FIGURES_DIR / "ablation_study.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] saved {out_path}")


def generate_all():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution()
    plot_learning_curves()
    plot_confusion_matrix()
    plot_ablation_bars()


if __name__ == "__main__":
    generate_all()
