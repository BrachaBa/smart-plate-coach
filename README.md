# Plate Coach Vision

Deep Learning component of **Smart Plate Coach AI** (see `docs/Smart_Plate_Coach_AI_Spec.pdf`, Part א׳):
a food-image classifier trained on Food-101 via Transfer Learning + Fine-Tuning, with a
Data Cleaning / Augmentation / Ablation Study pipeline that feeds the project's LLM Agent
(Part ב׳, not in this repo yet).

## Project layout

```text
data/food41/            Food-101-style dataset: images/, meta/meta/{classes,train,test}.txt
notebooks/               Exploratory data-quality notebook (sample-based detection only)
vision/
  dataset.py             Load splits, build class_to_idx, stratified train/val split, PyTorch Dataset
  vision_model.py         EfficientNet-B0 model builder, inference & train-time transforms
  data_cleaning.py         Full-dataset cleaning: corrupted/low-res/blank filtering, dedup, leakage removal
  train.py                 Core training loop; trains Ablation config (1) Baseline when run directly
  ablation.py              Ablation configs (2)-(4): cleaning, fine-tuning, augmentation
  checkpoints/             Saved model weights (*.pt) — created on first run
  results/                 Metrics/history CSVs and JSON — created on first run
requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Data is expected at `data/food41/` with `images/<class>/<id>.jpg` and `meta/meta/{classes,train,test}.txt`
(the standard Food-101 layout; this project's copy has 90 classes).

## Training

See `TRAINING.md` for the full step-by-step commands and expected runtimes.

Quick start (baseline only):

```bash
python -m vision.train
```

## Where things are documented

- **Product/requirements spec**: `docs/Smart_Plate_Coach_AI_Spec.pdf`
- **Deep Learning assignment brief**: `docs/main/מטלת סיכום מעשית ב-Deep Learning.pdf`
- **Training walkthrough (commands, runtimes)**: `TRAINING.md`
- **How the pipeline works and why (concepts)**: `HOW_IT_WORKS.md`
