# Training Walkthrough

Run these in order from the project root, with the venv active:

```bash
source venv/bin/activate
```

Each step below is a separate, long-running process — run each one yourself in your
terminal rather than chaining them, so you can monitor/interrupt individually.

## 0. Sanity check (optional, ~10s)

Confirms your device (MPS/CUDA/CPU) and data paths work before committing to a full run.

```bash
python -c "
from vision.dataset import load_food101_splits
classes, train_df, test_df = load_food101_splits('data/food41')
print(len(classes), 'classes,', len(train_df), 'train,', len(test_df), 'test')
"
```

## 1. Baseline (Ablation config 1)

Raw, uncleaned data, frozen EfficientNet-B0 backbone, no augmentation, 5 epochs.

```bash
python -u -m vision.train
```

- `-u` disables Python's stdout buffering so per-epoch progress prints immediately
  (without it, output can sit in a buffer and not appear until the process exits).
- Runs on ~60,750 train / 6,750 val images, then evaluates on 22,500 test images.
- **Estimated runtime: ~1–1.5 hours** on Apple Silicon (MPS). No mid-run checkpointing —
  if interrupted, it restarts from scratch.
- Outputs: `vision/checkpoints/baseline.pt`, `vision/results/baseline_history.csv`,
  `vision/results/baseline_metrics.json`.

To run in the background instead of keeping a terminal open:

```bash
nohup python -u -m vision.train > logs/baseline_train.log 2>&1 &
tail -f logs/baseline_train.log      # watch progress
```

Sleeping the machine is safe — the process suspends and resumes on wake. A full
shutdown kills it and loses all progress (no checkpoint exists until the run
finishes).

## 2. Data Cleaning

Opens and hashes every image in the dataset (~90k files): flags corrupted files,
too-low-resolution images, near-blank images, train→test leakage, and in-train
near-duplicates, then removes them.

```bash
python -u -m vision.data_cleaning
```

- **Estimated runtime: tens of minutes** (scanning + hashing every image on disk).
- Outputs: `vision/results/train_clean.csv`, `vision/results/test_clean.csv`,
  `vision/results/cleaning_report.json`.
- Independent of step 1 — can run before, during (if you have CPU headroom), or
  after the baseline.

## 3. Ablation configs 2–4

Requires step 2's cleaned CSVs to exist first.

```bash
python -u -m vision.ablation
```

This runs, in order:

- **Config 2** (+ cleaning): cleaned data, frozen backbone, no augmentation.
- **Config 3** (+ fine-tuning): cleaned data, Transfer Learning → Fine-Tuning (backbone
  unfreezes after the transfer phase, at a lower learning rate).
- **Config 4** (+ augmentation, full pipeline): same as config 3, plus train-only
  Flip/Rotation/Color-Jitter augmentation.
- A final summary comparing all 4 configs' test accuracy/F1, written to
  `vision/results/ablation_summary.csv`. This also checks the spec's success
  criterion: config 4 must beat config 1 (baseline) by ≥15% on Accuracy or F1.

Each config trains and evaluates independently, so **this is the longest step**
— expect it to take at least as long as the baseline, likely longer (config 3
and 4 each run two training phases). Run it in the background the same way as
step 1 if you don't want to keep a terminal open.

To run a single config manually (e.g. to retry just one after a crash):

```bash
python -c "from vision.ablation import run_config_2_cleaning; run_config_2_cleaning()"
python -c "from vision.ablation import run_config_3_finetuning; run_config_3_finetuning()"
python -c "from vision.ablation import run_config_4_full; run_config_4_full()"
python -c "from vision.ablation import summarize_ablation; summarize_ablation()"
```

## Adjusting epochs/batch size

All entry points take keyword overrides, e.g.:

```bash
python -c "from vision.train import run_baseline; run_baseline(epochs=3, batch_size=32)"
```

Lower `epochs` and/or `batch_size` for a faster (less accurate) smoke test before
committing to a full run.

## Checking progress without waiting

```bash
ps aux | grep vision                          # is it still running?
ls -la vision/checkpoints vision/results      # files appear when a run finishes
cat vision/results/baseline_metrics.json      # once it exists, that run is done
```
