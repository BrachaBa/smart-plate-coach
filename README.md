
# Smart Plate Coach AI

Two parts, per `docs/Smart_Plate_Coach_AI_Spec.pdf`:

- **Part א׳ — Vision**: a food-image classifier trained on Food-101 via Transfer Learning +
  Fine-Tuning, with a Data Cleaning / Augmentation / Ablation Study pipeline. **Done.**
  Full pipeline (cleaning + fine-tuning + augmentation) reached **82.8% test accuracy**
  (0.827 macro-F1), a **35.6% improvement** over the 61.1% baseline — well past the spec's
  required 15%. See `HOW_IT_WORKS.md`.
- **Part ב׳ — Agent**: a LangGraph + OpenRouter LLM agent ("Plate Coach") that consumes the
  vision model's output to analyze meal photos, compute a Healthy Score, run what-if
  simulations, and give general (non-medical) suggestions, via an interactive CLI.
  **Core + CLI done; evals and packaging (manifest.json, PyInstaller) still pending.**
  See `HOW_AGENT_WORKS.md`.

## Project layout

```text
data/food41/            Food-101-style dataset: images/, meta/meta/{classes,train,test}.txt
notebooks/               Exploratory data-quality notebook (sample-based detection only)
logs/                    Training stdout logs when run in the background (not committed)
vision/
  dataset.py             Load splits, build class_to_idx, stratified train/val split, PyTorch Dataset
  vision_model.py         EfficientNet-B0 model builder, inference & train-time transforms
  data_cleaning.py         Full-dataset cleaning: corrupted/low-res/blank filtering, dedup, leakage removal
  train.py                 Core training loop; trains Ablation config (1) Baseline when run directly
  ablation.py              Ablation configs (2)-(4): cleaning, fine-tuning, augmentation
  visualizations.py        Generates the 4 required plots into vision/results/figures/
  checkpoints/             Saved model weights (*.pt) — created on first run, not committed
  results/                 Metrics/history CSVs, JSON, and figures/ — created on first run
agent/
  class_names.py          Static 90 Food-101 class names, matching the trained checkpoint
  vision_model.py         Inference wrapper around vision/vision_model.py + the trained checkpoint
  tools.py                RAG nutrition lookup + Code-Exec Healthy Score / what-if delta functions
  data/nutrition_facts.json  Local nutrition-facts table (RAG source, ~50 foods)
  state.py                Shared LangGraph state schema
  nodes.py                All 10 graph nodes + the scope guard
  graph.py                Compiled LangGraph workflows (analysis, simulation, history)
  guardrails.py           Prescriptive-language rewriting + disclaimer
  llm.py                  OpenRouter client wiring (models configurable via .env)
  pricing.py              Placeholder per-model token pricing for cost display/evals
  cli.py                  Interactive CLI entrypoint (python -m agent.cli)
evals/                    LLM-as-judge / cost / scope evals (not yet implemented)
requirements.txt
.env.example              Required env vars (OpenRouter, LangSmith) — no real values
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # agent only
# pip install -r requirements-vision.txt # + training the vision model (torch)
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

Data is expected at `data/food41/` with `images/<class>/<id>.jpg` and `meta/meta/{classes,train,test}.txt`
(the standard Food-101 layout; this project's copy has 90 classes).

## Training the vision model

See `TRAINING.md` for the full step-by-step commands and expected runtimes.

Quick start (baseline only):

```bash
python -m vision.train
```

## Running the agent

```bash
python -m agent.cli
```

The trained classifier ships in the repo as `agent/models/food_classifier.onnx` (exported from
`config4_full.pt` by `python -m scripts.export_onnx`), so no training is needed. It needs
`OPENROUTER_API_KEY` set in `.env`. See `RUNNING_AGENT.md` for a full walkthrough and
troubleshooting (including an SSL note relevant to networks with a TLS-intercepting proxy).

## Where things are documented

- **Product/requirements spec**: `docs/Smart_Plate_Coach_AI_Spec.pdf`
- **Deep Learning assignment brief**: `docs/main/מטלת סיכום מעשית ב-Deep Learning.pdf`
- **LLM Agent assignment brief**: `docs/main/פרויקט_סיום_LLMs.docx`
- **Vision training walkthrough (commands, runtimes)**: `TRAINING.md`
- **How the vision pipeline works and why (concepts)**: `HOW_IT_WORKS.md`
- **Agent walkthrough (commands, example session, troubleshooting)**: `RUNNING_AGENT.md`
- **How the agent works and why (concepts)**: `HOW_AGENT_WORKS.md`
