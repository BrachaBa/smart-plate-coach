# Running the Agent

Run these from the project root, with the venv active:

```bash
source venv/bin/activate
```

## 0. Prerequisites

- The vision model must already be trained: `vision/checkpoints/config4_full.pt` must exist
  (see `TRAINING.md` — specifically the ablation step, config 4). The agent's vision intake
  node loads this checkpoint directly and will raise `FileNotFoundError` if it's missing.
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

You'll see the graph's progress streamed node-by-node:

```text
[INFO] Entering Node: scope_guard...
[INFO] Entering Node: vision_intake...
[INFO] Entering Node: input_gate...
[INFO] Entering Node: nutrition_analysis...
[INFO] Entering Node: scoring...
[INFO] Entering Node: improvement_planner...
[INFO] Entering Node: goal_adjuster...
[INFO] Entering Node: evaluator...
[INFO] Entering Node: hitl_gate...

Healthy Score: 66/100
Breakdown: {'protein': 60, 'veg': 15, 'carbs': 100, 'sugar': 70, 'sat_fat': 92}
Profile: protein: moderate, carbs: high, veg: low, sugar: moderate

Suggestions:
  - Add a side of steamed broccoli or mixed vegetables (...) (projected: 73/100)
  - ...

This is general lifestyle information only, not professional medical or dietary advice. ...
[cost] 506 prompt + 166 completion tokens (~$0.0000, theoretical $0.0040)
```

If the image can't be classified confidently, the input gate will say so and ask for a
clearer photo instead of guessing.

## 3. Ask "what if" questions

Right after an analysis, you're dropped into a what-if loop:

```text
what-if> what if I add a side salad
  Before: score 66, 375.0 kcal
  After:  score 73, 412.5 kcal
  Delta:  {'score': 7, 'calories': 37.5, 'protein_g': 2.2, 'carbs_g': 6.0, 'sugar_g': 3.0, 'saturated_fat_g': 0.0}
```

Press Enter on a blank line to return to the main prompt, or type `quit`/`exit` to leave
Plate Coach entirely from within the what-if loop. Off-topic questions asked here are refused
the same way they would be at the main prompt.

## 4. Other commands

```text
help  or  ?          Show usage help
history               Enter up to 20 meal-photo paths for a weekly trend summary
export-summary        Write the last analysis's final output to plate_coach_summary.txt
quit  or  exit         Exit
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

**`openai.NotFoundError: ... model is unavailable for free ... use this slug instead`**

The free-tier model slug in `.env` is no longer valid — OpenRouter's free model catalog
changes over time. Pick a current one from
[openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0) and update
`DEFAULT_MODEL`/`JUDGE_MODEL` in `.env`.

**`FileNotFoundError: Vision checkpoint not found at .../config4_full.pt`**

Run the vision training pipeline first — see `TRAINING.md`, specifically
`python -u -m vision.ablation` (config 4 is the full pipeline checkpoint the agent loads).

**A suggestion shows no `projected:` score**

This is expected, not a bug: the improvement planner only attaches a projected score when the
food it's proposing to change is one that's actually part of the current meal (for
remove/swap) or resolvable in the local nutrition database (for add). If the LLM proposes a
change to something outside that, no number is shown rather than a fabricated one.
