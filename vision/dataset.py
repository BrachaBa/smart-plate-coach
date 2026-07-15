"""Food-101 metadata loading and a PyTorch Dataset over the image files.

This is the single source of truth for building train/test DataFrames — the
notebook's Task 2 cells and any future training script should both call
`load_food101_splits` instead of duplicating the loading logic.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_food101_splits(data_dir: Path):
    """Read Food-101 metadata files and return (classes, train_df, test_df).

    Each returned DataFrame has columns: id, class, path.
    """
    data_dir = Path(data_dir)
    meta_dir = data_dir / "meta" / "meta"
    img_dir = data_dir / "images"

    classes = pd.read_csv(meta_dir / "classes.txt", header=None, names=["class"])
    train_ids = pd.read_csv(meta_dir / "train.txt", header=None, names=["id"])
    test_ids = pd.read_csv(meta_dir / "test.txt", header=None, names=["id"])

    def to_df(ids: pd.DataFrame) -> pd.DataFrame:
        df = ids.copy()
        df["class"] = df["id"].str.split("/").str[0]
        df["path"] = df["id"].apply(lambda x: img_dir / f"{x}.jpg")
        return df

    return classes, to_df(train_ids), to_df(test_ids)


def make_class_to_idx(classes: pd.DataFrame) -> dict:
    """One shared class->int mapping, derived from classes.txt (not per-split)."""
    return {c: i for i, c in enumerate(classes["class"])}


def split_train_val(train_df: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42):
    """Stratified train/val split: val_fraction of each class goes to validation.

    Stratifying per-class (rather than a single random split of all rows)
    guarantees every class is represented in validation even with 90 classes,
    and keeps the split reproducible across every training script that needs
    it (baseline, fine-tune, all 4 ablation configs) via the fixed seed.
    """
    val_parts = []
    train_parts = []
    for _, group in train_df.groupby("class"):
        val_group = group.sample(frac=val_fraction, random_state=seed)
        train_group = group.drop(val_group.index)
        val_parts.append(val_group)
        train_parts.append(train_group)

    val_df = pd.concat(val_parts).reset_index(drop=True)
    train_df = pd.concat(train_parts).reset_index(drop=True)
    return train_df, val_df


class FoodImageDataset(Dataset):
    """Wraps a train/test DataFrame (id, class, path columns) for PyTorch.

    class_to_idx must be passed in explicitly (rather than derived per-split)
    so train/val/test all agree on the same integer label for a given class.
    """

    def __init__(self, dataframe: pd.DataFrame, class_to_idx: dict, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["class"]]
        return image, label
