# Running the Agent

Run these from the project root, with the venv active:

```bash
source venv/bin/activate
```

## 0. Prerequisites

- Nothing to train: the classifier ships as `agent/models/food_classifier.onnx` and runs with
  `onnxruntime` (no torch). It was exported from `vision/checkpoints/config4_full.pt` by
  `python -m scripts.export_onnx`; re-run that only if you retrain the model.
- An OpenRouter API key. Sign up at [openrouter.ai](https://openrouter.ai), generate a key,
  and put it in `.env`:

  ```bash
  cp .env.example .env
  # edit .env, set OPENROUTER_API_KEY=<your key>
  ```

  `DEFAULT_MODEL`/`PAID_MODEL`/`JUDGE_MODEL` in `.env.example` are placeholder model slugs.
  OpenRouter's free-tier model list changes over time — if you get a 404 error mentioning a
  model slug, check [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0)
  for a currently-available free model and update `.env`.

## 1. Start the CLI

```bash
python -m agent.cli
```

You'll be greeted with a short help message, then asked to pick a session goal
(weight loss / maintain / balanced / muscle gain) — this only affects the *wording* of
suggestions, never the computed Healthy Score.

## 2. Analyze a meal photo

At the `plate-coach>` prompt, give a path to a local image:

```text
plate-coach> data/food41/images/sushi/100332.jpg
```

You'll see the graph's progress streamed node-by-node, as each node starts:

```text
[INFO] Entering Node: vision_intake...
[INFO] Entering Node: input_gate...
[INFO] Entering Node: nutrition_analysis...
[INFO] Entering Node: scoring...
[INFO] Entering Node: improvement_planner...
[INFO] Entering Node: goal_adjuster...
[INFO] Entering Node: evaluator...
[INFO] Entering Node: propose_save...
[INFO] Entering Node: hitl_gate...

[HITL] The agent wants to save this meal (sushi, score 66) to your meal history:
       C:\...\plate_coach_history.json
Proceed? [Y/N]: y
[INFO] Resuming Node: hitl_gate (approved)...
```

Then the result -- the identified dish, the Healthy Score with its breakdown, a qualitative
profile, 2-4 suggestions each with the projected score computed by code, the disclaimer -- and a
per-node cost table (`[COST]`) with the actual and the theoretical cost.

If the image can't be classified confidently, the input gate says so (with its best guesses)
and asks for a clearer photo instead of guessing. Answering `N` at the `[HITL]` prompt saves
nothing.

## 3. Ask "what if" questions

Right after an analysis, you're dropped into a what-if loop:

```text
what-if> what if I add a side salad
[INFO] Entering Node: scope_guard...
[INFO] Entering Node: simulation_handler...
  Before: score 66, 375.0 kcal
  After:  score 73, 412.5 kcal
  Change: score +7, calories +37.5, protein +2.2 g, sugar +3.0 g
```

Press Enter on a blank line to return to the main prompt, or type `quit`/`exit` to leave.
Off-topic questions asked here are refused the same way as at the main prompt.

## 4. Questions and other commands

Any text that is not a command or an image path is treated as a nutrition question
(`how much protein is in sushi?`). It goes through the scope guard first; an off-topic or
prompt-injection request is refused.

```text
history          Up to 20 photo paths (or Enter to use your saved meals) -> trend summary
export-summary   Save a summary of the conversation to plate_coach_summary_<time>.txt ([Y/N] first)
model [name]     Show or switch the OpenRouter model (e.g. model openai/gpt-4o-mini)
status           Model, API key and LangSmith tracing status
cost             Session total tokens and cost
help  or  ?      Show usage help
quit  or  exit   Exit
```

## Troubleshooting

**`openai.APIConnectionError: Connection error` / `SSL: CERTIFICATE_VERIFY_FAILED`**

Some networks (school/ISP/corporate content filters) run a TLS-intercepting proxy. Its root
certificate is trusted by your OS but may not be in Python's bundled `certifi` CA list, or may
be missing an X.509 extension that Python 3.13+'s stricter verification requires — even though
`curl`/your browser connect fine. `agent/llm.py` already works around this by using the
`truststore` package, which makes Python's SSL use the OS-native trust store instead of
`certifi`'s bundled one. If you still see this error, confirm `truststore` is installed
(`pip show truststore`) and that you're on a reasonably current OS trust store.

**`[ERROR] Rate limit reached ...`**

Free models are often rate-limited upstream. The agent already falls back to the next model in
`FREE_FALLBACK_MODELS`; if all of them are busy, wait a minute or switch to a paid model
(`model openai/gpt-4o-mini`).

**`[ERROR] OpenRouter does not know the model ...`**

The free-tier model slug in `.env` is no longer valid — OpenRouter's free model catalog
changes over time. Pick a current one from
[openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0) and update
`DEFAULT_MODEL`/`JUDGE_MODEL` in `.env`.

**`FileNotFoundError: Vision model not found at .../food_classifier.onnx`**

The ONNX file is missing from your checkout. Pull it again, or regenerate it from a trained
checkpoint: `pip install -r requirements-vision.txt && python -m scripts.export_onnx`.

**A suggestion shows no `projected:` score**

This is expected, not a bug: the improvement planner only attaches a projected score when the
food it's proposing to change is one that's actually part of the current meal (for
remove/swap) or resolvable in the local nutrition database (for add). If the LLM proposes a
change to something outside that, no number is shown rather than a fabricated one.
