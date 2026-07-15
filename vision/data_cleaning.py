"""Full-dataset data cleaning (spec 2.2): mislabeled review, duplicate/leakage
removal, and corrupted/outlier filtering.

The notebook (notebooks/data-cleaning.ipynb) only *detects* these issues on a
2000-image sample. This module scales detection to the full train/test sets
and actually removes what should be removed, producing a cleaned train_df
that `vision/train.py` can consume for ablation configs 2-4.
"""

import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "food41"
RESULTS_DIR = PROJECT_ROOT / "vision" / "results"

LOW_RES_MIN_DIM = 128  # spec 2.2: too low resolution for a 224x224 EfficientNet input
NEAR_BLANK_STD = 5.0  # grayscale std below this ~= near-uniform/blank image
DUPLICATE_HAMMING_DISTANCE = 5  # near-identical phash threshold
HASH_BUCKET_PREFIX_BITS = 16  # bucket by top bits of the hash to avoid O(n^2) comparison


def _is_corrupted(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return False
    except (UnidentifiedImageError, OSError):
        return True


def _image_stats(path: Path):
    """Return (min_dim, grayscale_std) or (None, None) on failure."""
    try:
        with Image.open(path) as im:
            width, height = im.size
            gray = np.array(im.convert("L"))
        return min(width, height), float(gray.std())
    except (UnidentifiedImageError, OSError):
        return None, None


def _phash(path: Path):
    try:
        return imagehash.phash(Image.open(path))
    except (UnidentifiedImageError, OSError):
        return None


def flag_corrupted_and_outliers(df: pd.DataFrame, progress_every: int = 5000) -> pd.DataFrame:
    """Annotate df (id, class, path, ...) with is_corrupted / min_dim / std_px.

    Runs PIL.verify + resolution/brightness checks over every row. This opens
    every image on disk once — expect this to take a while over tens of
    thousands of images.
    """
    df = df.copy()
    is_corrupted = []
    min_dims = []
    std_pxs = []

    for i, path in enumerate(df["path"]):
        if progress_every and i and i % progress_every == 0:
            print(f"[cleaning] scanned {i}/{len(df)} images")

        if not Path(path).exists():
            is_corrupted.append(True)
            min_dims.append(None)
            std_pxs.append(None)
            continue

        corrupted = _is_corrupted(path)
        is_corrupted.append(corrupted)
        if corrupted:
            min_dims.append(None)
            std_pxs.append(None)
        else:
            min_dim, std_px = _image_stats(path)
            min_dims.append(min_dim)
            std_pxs.append(std_px)

    df["is_corrupted"] = is_corrupted
    df["min_dim"] = min_dims
    df["std_px"] = std_pxs
    df["is_low_res"] = df["min_dim"].apply(lambda d: d is not None and d < LOW_RES_MIN_DIM)
    df["is_near_blank"] = df["std_px"].apply(lambda s: s is not None and s < NEAR_BLANK_STD)
    return df


def compute_phashes(df: pd.DataFrame, progress_every: int = 5000) -> pd.DataFrame:
    """Add a 'phash' column (perceptual hash) to df. Skips rows already flagged corrupted."""
    df = df.copy()
    hashes = []
    for i, row in enumerate(df.itertuples()):
        if progress_every and i and i % progress_every == 0:
            print(f"[hashing] hashed {i}/{len(df)} images")
        if getattr(row, "is_corrupted", False):
            hashes.append(None)
        else:
            hashes.append(_phash(row.path))
    df["phash"] = hashes
    return df


def _bucket_key(h: imagehash.ImageHash) -> int:
    """Top HASH_BUCKET_PREFIX_BITS bits of the hash, used to avoid full O(n^2) comparison."""
    return int(str(h), 16) >> (64 - HASH_BUCKET_PREFIX_BITS)


def find_duplicate_indices(df: pd.DataFrame, max_distance: int = DUPLICATE_HAMMING_DISTANCE) -> set:
    """Indices of df whose phash is a near-duplicate of an earlier row in df.

    Only compares within hash buckets sharing the same top bits — true
    O(n^2) over the full ~90k images is too slow, per the notebook's note.
    """
    hashed = df.dropna(subset=["phash"])
    buckets: dict[int, list] = {}
    for idx, h in hashed["phash"].items():
        buckets.setdefault(_bucket_key(h), []).append((idx, h))

    duplicate_indices = set()
    for group in buckets.values():
        for i in range(len(group)):
            idx_i, hash_i = group[i]
            if idx_i in duplicate_indices:
                continue
            for j in range(i + 1, len(group)):
                idx_j, hash_j = group[j]
                if hash_i - hash_j <= max_distance:
                    duplicate_indices.add(idx_j)
    return duplicate_indices


def find_leaked_test_indices(
    train_df: pd.DataFrame, test_df: pd.DataFrame, max_distance: int = DUPLICATE_HAMMING_DISTANCE
) -> set:
    """Train indices whose phash near-matches a test image (train/test leakage)."""
    test_hashed = test_df.dropna(subset=["phash"])
    test_buckets: dict[int, list] = {}
    for h in test_hashed["phash"]:
        test_buckets.setdefault(_bucket_key(h), []).append(h)

    leaked = set()
    for idx, h in train_df.dropna(subset=["phash"])["phash"].items():
        for test_hash in test_buckets.get(_bucket_key(h), []):
            if h - test_hash <= max_distance:
                leaked.add(idx)
                break
    return leaked


def clean_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Full spec-2.2 cleaning pass. Returns (cleaned_train_df, cleaned_test_df, report).

    Order: flag corrupted/low-res/near-blank on both splits -> drop those rows
    -> hash survivors -> drop train images that leak into test -> drop
    in-train near-duplicates (keep first occurrence per class order).
    """
    train_df = flag_corrupted_and_outliers(train_df)
    test_df = flag_corrupted_and_outliers(test_df)

    train_bad = train_df["is_corrupted"] | train_df["is_low_res"] | train_df["is_near_blank"]
    test_bad = test_df["is_corrupted"] | test_df["is_low_res"] | test_df["is_near_blank"]

    train_clean = train_df[~train_bad].reset_index(drop=True)
    test_clean = test_df[~test_bad].reset_index(drop=True)

    train_clean = compute_phashes(train_clean)
    test_clean = compute_phashes(test_clean)

    leaked = find_leaked_test_indices(train_clean, test_clean)
    train_clean = train_clean.drop(index=leaked).reset_index(drop=True)

    dupes = find_duplicate_indices(train_clean)
    train_clean = train_clean.drop(index=dupes).reset_index(drop=True)

    report = {
        "train_total": len(train_df),
        "train_corrupted": int(train_df["is_corrupted"].sum()),
        "train_low_res": int(train_df["is_low_res"].sum()),
        "train_near_blank": int(train_df["is_near_blank"].sum()),
        "train_leaked_into_test_removed": len(leaked),
        "train_duplicates_removed": len(dupes),
        "train_remaining": len(train_clean),
        "test_total": len(test_df),
        "test_corrupted": int(test_df["is_corrupted"].sum()),
        "test_low_res": int(test_df["is_low_res"].sum()),
        "test_near_blank": int(test_df["is_near_blank"].sum()),
        "test_remaining": len(test_clean),
    }

    return (
        train_clean.drop(columns=["is_corrupted", "min_dim", "std_px", "is_low_res", "is_near_blank", "phash"]),
        test_clean.drop(columns=["is_corrupted", "min_dim", "std_px", "is_low_res", "is_near_blank", "phash"]),
        report,
    )


def flag_mislabel_candidates(
    df: pd.DataFrame,
    model,
    class_to_idx: dict,
    transform,
    device,
    confidence_threshold: float = 0.9,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Flag likely-mislabeled rows using an already-trained model's confidence
    (spec 2.2: "Confidence-based" detection). Returns a review DataFrame with
    predicted_class/confidence columns — for manual review, not auto-removal,
    since only a human can confirm a label is actually wrong.
    """
    from torch.utils.data import DataLoader

    from vision.dataset import FoodImageDataset

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    ds = FoodImageDataset(df, class_to_idx, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    predicted_classes, confidences = [], []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            top_prob, top_idx = probs.max(dim=1)
            predicted_classes.extend(top_idx.cpu().tolist())
            confidences.extend(top_prob.cpu().tolist())

    review = df.reset_index(drop=True).copy()
    review["predicted_class"] = [idx_to_class[i] for i in predicted_classes]
    review["confidence"] = confidences
    mismatched = review[
        (review["predicted_class"] != review["class"]) & (review["confidence"] >= confidence_threshold)
    ]
    return mismatched.sort_values("confidence", ascending=False)


def run_cleaning():
    """Clean the full train/test sets and write cleaned CSVs + a report to disk.

    Opens and hashes every image under data/food41/images (~90k files) —
    expect this to take a while. Outputs:
      vision/results/train_clean.csv, test_clean.csv, cleaning_report.json
    """
    from vision.dataset import load_food101_splits

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, train_df, test_df = load_food101_splits(DATA_DIR)

    print(f"[cleaning] starting: train={len(train_df)} test={len(test_df)}")
    train_clean, test_clean, report = clean_dataset(train_df, test_df)

    train_clean.to_csv(RESULTS_DIR / "train_clean.csv", index=False)
    test_clean.to_csv(RESULTS_DIR / "test_clean.csv", index=False)
    with open(RESULTS_DIR / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[cleaning] done:", report)
    return train_clean, test_clean, report


if __name__ == "__main__":
    run_cleaning()
