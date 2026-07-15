# How the Vision Pipeline Works

This is the conceptual companion to `TRAINING.md` (which just has commands). Read this
if you want to understand *why* the pipeline is built this way, not just how to run it.

## The task

Classify a photo of a meal into one of 90 food classes (Food-101-style dataset in
`data/food41/`), as the first stage of the larger Smart Plate Coach AI product
(`docs/Smart_Plate_Coach_AI_Spec.pdf`). The LLM Agent (Part ב׳ of the spec) consumes
this classifier's output — it is not built in this repo yet.

## Why EfficientNet-B0, and why pretrained

Training a CNN from random weights to recognize food would need far more labeled
images and far more compute than this project has. Instead, `vision_model.py` starts
from **EfficientNet-B0 pretrained on ImageNet** (1000 general object classes) and
replaces only the final classification layer with a fresh 90-way head. The backbone
already knows how to detect edges, textures, and shapes — food images share those
low-level visual patterns with everyday objects. This is called **Transfer Learning**.

## Transfer Learning, then Fine-Tuning — why two phases

Two phases, run in sequence (`vision_model.set_backbone_trainable`, orchestrated by
`train.train_transfer_then_finetune`):

1. **Transfer Learning phase**: the backbone is frozen (`freeze_backbone=True`) —
   only the new classifier head trains. This is fast and safe: with a randomly
   initialized head sitting on top of a pretrained backbone, early gradients would
   otherwise be large and noisy enough to damage the backbone's already-good weights.
2. **Fine-Tuning phase**: once the head has learned something reasonable, the backbone
   unfreezes (`freeze_backbone=False`) and trains too, at a **lower learning rate**
   (`finetune_lr=1e-4` vs `transfer_lr=1e-3` by default). This lets the backbone's
   features specialize toward food images specifically (e.g. telling rice apart from
   quinoa, which ImageNet never needed to distinguish) — without a learning rate that's
   large enough to erase what it already knows.

Skipping straight to fine-tuning (no frozen phase first) is why config 2 in the
ablation study — cleaning only, no fine-tuning — exists as a checkpoint: it isolates
how much of the improvement comes from fine-tuning specifically, versus just having
cleaner data.

## Why clean the data at all

Raw datasets scraped from the wild have real defects that hurt training in different
ways (`data_cleaning.py`, spec section 2.2):

| Defect | Why it hurts | How it's caught |
|---|---|---|
| Mislabeled images | Model learns the wrong association | Manual review helper (`flag_mislabel_candidates`) — flags for a human, never auto-removes, since only a human can confirm a label is actually wrong |
| Corrupted files | Crashes the data loader mid-training | `PIL.Image.verify()` on every file |
| Near-duplicate images across train/test | The model "memorizes" a test image it already saw in train — inflates test accuracy without real generalization (data leakage) | Perceptual hashing (`imagehash.phash`), compared train vs test |
| Near-duplicate images within train | Wastes training time re-learning the same image, can bias the class balance | Perceptual hashing, compared within train |
| Too-low resolution / near-blank images | No usable signal for the model to learn from | Measuring pixel dimensions and grayscale standard deviation |

Perceptual hashing (rather than exact byte match) catches images that are visually
the same but re-encoded, resized, or slightly cropped — exact duplicates are rare in
practice, near-duplicates are not.

**Why hash by bucket instead of comparing every pair:** comparing every image against
every other image is O(n²) — at ~90,000 images that's ~4 billion comparisons, too slow
to run. `data_cleaning.py` buckets images by the top bits of their hash first, and only
compares images within the same bucket. Two images with a small Hamming distance
between their hashes will usually share those top bits, so this catches the same
duplicates in a fraction of the time — at the cost of a small chance of missing a
duplicate pair that happens to fall in different buckets.

## Why augmentation is train-only

`vision_model.get_train_transforms()` adds random Flip, Rotation (±15°), Color-Jitter,
and a light Random Crop — but only to the training set (spec section 2.3). The
model never augments what it's evaluated on: if it did, the reported accuracy
would reflect a made-up, easier version of the test set, not real-world performance.

The point of augmentation is to simulate the *variety* a real phone photo will have —
different lighting, angles, and framing — so the model doesn't overfit to the
specific lighting/angle/framing of the training photos it happened to see. More
variety during training, same real photos during evaluation.

## The Ablation Study — why 4 configs instead of one

Training one final model and reporting its accuracy tells you *that* it works, not
*why*. The ablation study (`ablation.py`, spec section 2.4) isolates each
intervention's individual contribution by adding exactly one thing at a time:

1. **Baseline** (`train.py`): raw uncleaned data, frozen backbone, no augmentation.
   The floor — what you get from transfer learning alone, with no other effort.
2. **+ Cleaning**: same frozen backbone and no augmentation, but on cleaned data.
   Isolates how much cleaning alone is worth.
3. **+ Fine-Tuning**: cleaned data, now with the two-phase transfer→fine-tune
   training. Isolates fine-tuning's contribution on top of cleaning.
4. **+ Augmentation** (full pipeline): everything in config 3, plus train-time
   augmentation. This is the model you'd actually ship.

Comparing configs 1→2→3→4 in sequence tells you which intervention mattered most —
useful for deciding where to spend more effort if the final result isn't good enough,
rather than guessing.

## The 15% success criterion

The spec requires config 4 to beat config 1 (baseline) by at least 15% on Accuracy or
F1 (macro-averaged, so all 90 classes count equally regardless of how many images each
has). `ablation.summarize_ablation()` computes this automatically once all 4 configs'
metrics exist. If it's not met, the ablation breakdown (which config gained the most)
is the starting point for deciding what to strengthen — more epochs, a stronger
augmentation policy, or revisiting the cleaning thresholds are the usual first things
to check.

## How to read the results once they exist

- `vision/results/*_history.csv` — per-epoch train/val loss and accuracy for each
  config. Plot train vs val over epochs (Learning Curves, spec 2.4): a widening gap
  between them is overfitting.
- `vision/results/*_metrics.json` — each config's final test accuracy and macro-F1.
- `vision/results/ablation_summary.csv` — all 4 configs side by side, plus the 15%
  check.
- `vision/results/cleaning_report.json` — how many images were removed and why
  (corrupted / low-res / near-blank / leaked / duplicate), and how many remain.

## Where the code lives, mapped to the concepts above

- `vision/dataset.py` — loading Food-101 metadata, stratified train/val split,
  the PyTorch `Dataset` wrapper.
- `vision/vision_model.py` — EfficientNet-B0 construction, freeze/unfreeze logic,
  inference vs train-time transforms.
- `vision/data_cleaning.py` — corrupted/low-res/blank filtering, perceptual-hash
  dedup and leakage removal, mislabel-review helper.
- `vision/train.py` — the shared train/eval loop (`train_model`, `evaluate_on_test`),
  the transfer→fine-tune orchestration (`train_transfer_then_finetune`), and the
  baseline entry point (`run_baseline`).
- `vision/ablation.py` — configs 2–4 built from the same `train_model` building
  blocks, plus the final comparison/success-criterion check.
