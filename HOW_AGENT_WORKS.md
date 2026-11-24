# How the Agent Works

This is the conceptual companion to `RUNNING_AGENT.md` (which just has commands). Read this
if you want to understand *why* the agent is built this way, not just how to run it.

## The task

Plate Coach is a single-purpose LLM agent (`docs/Smart_Plate_Coach_AI_Spec.pdf`, Part ב׳):
given a meal photo, classify the food (using the Part א׳ vision model), analyze its nutrition,
compute a Healthy Score, propose small improvements, answer "what if" questions, and track
weekly trends across meals. It is explicitly **not** a general chatbot — it refuses anything
outside meal-nutrition analysis, and every response carries a disclaimer that it's general
lifestyle information, not medical or dietary advice.

## Why the LLM never computes a number

Every numeric result in the agent — the Healthy Score, calorie/macro totals, the projected
score after a suggested change, the before/after delta for a "what if" question — is computed
by plain, deterministic Python in `agent/tools.py`, never by the LLM doing arithmetic in prose.
This is the spec's "Code-Exec" pattern: the LLM decides *what* changed (which food, which
action), and `agent/tools.py` decides *what that means numerically*. LLMs are unreliable at
exact arithmetic and don't reliably reproduce the same number twice for the same input — for a
score the user might act on, that's not acceptable.

Concretely: `agent/nodes.py`'s `improvement_planner_node` asks the LLM for suggestions as
structured JSON (`{"action": "add", "food": null, "replacement": "side_salad"}`, etc.), then
calls `agent.tools.compute_delta` on that structure to get the real projected score. The LLM
never sees or reports a number it invented itself.

## Why the LLM's structured output still needs normalizing

Even with a clear JSON schema in the prompt, real models don't always follow it exactly —
during live testing, one model wrapped its JSON in ` ```json ` markdown fences (stripped by
`agent/nodes.py::_parse_json_response`), and another put the food being *added* into the
`"food"` field instead of `"replacement"` (corrected by `_normalize_change`). Both are handled
defensively rather than trusted at face value, because the alternative — a mis-parsed action
silently producing a wrong or zero-effect score — is worse than a small normalization step.

## The RAG nutrition database

`agent/data/nutrition_facts.json` is a local table of calories/macros/veg content per 100 g
for 105 foods: every one of the 90 dishes the classifier knows, plus common add-ons the planner
suggests (side salad, water, brown rice, ...). This is deliberately a plain JSON file, not a
vector database: the pattern is RAG because the LLM's answers are grounded in *retrieved*
records rather than invented ones, and at ~100 records a similarity index buys nothing an exact
lookup doesn't. Retrieval is exact on a normalized name (`find_food`: lowercase, articles
dropped, spaces to underscores, singular form); for free-text questions `find_foods_in_text`
finds every database food named in the question as whole words. There is deliberately no
substring matching -- it used to match `apple_pie` to `apple` and score a pie as fruit. A food
that is not in the table gets a neutral default profile and is flagged as unmatched; a
suggestion that would rely on such a food shows no projected score instead of a made-up one.
The values are typical figures per dish, not a measurement of a specific plate.

## The Healthy Score formula

`agent/tools.py::compute_healthy_score` is a weighted 0-100 score: protein (0.25), vegetable
content (0.20), a carb-excess penalty (0.15), a sugar penalty (0.20), and a saturated-fat
penalty (0.20), each sub-score clipped to [0, 100] before weighting so no single extreme macro
(e.g. a very high-protein meal) can push the total out of range on its own. The thresholds
(`PROTEIN_TARGET_G`, `SUGAR_LIMIT_G`, `SATFAT_LIMIT_G`, `CARB_LIMIT_G`) are general per-meal
reference figures, not a personalized medical target — the agent never tailors these to an
individual's body data, in line with the spec's ethical constraints (Part ד׳).

## The LangGraph workflow

`agent/graph.py` compiles five graphs from the same node module (`agent/nodes.py`):

- **`build_analysis_graph()`** -- the photo flow: `vision_intake -> input_gate ->
  nutrition_analysis -> scoring -> improvement_planner -> goal_adjuster -> evaluator ->
  propose_save -> hitl_gate`. The input gate short-circuits to `END` on a low-confidence photo
  ("please provide a clearer photo"); the evaluator can loop back to the planner once, passing
  its notes along, before it accepts what it has.
- **`build_question_graph()`** -- `scope_guard -> nutrition_qa`: a free-text nutrition
  question, answered from the retrieved database records.
- **`build_simulation_graph()`** -- `scope_guard -> simulation_handler`: a what-if question
  against the last analyzed meal, without re-running vision/RAG/scoring.
- **`build_history_graph()`** -- runs vision+RAG+scoring over up to 20 photos (or the saved
  meals) and asks the LLM only for the trend *summary*, not the trend *numbers*.
- **`build_export_graph()`** -- just the HITL gate, for exporting the conversation summary.

### The scope guard: two layers, fail closed

Every free-text path starts at `scope_guard_node`. First a deterministic check
(`guardrails.detect_prompt_injection`) blocks attempts to override the instructions ("ignore
previous instructions", "you are now...", "reveal your system prompt") with no LLM call at all.
Then a short LLM call classifies the topic; the user's text is wrapped in tags and the model is
told to treat it as data. Anything that is not clearly `INSCOPE` is refused, so an odd model
reply fails closed. A photo path does not go through it: submitting a photo is the task itself,
and the input gate checks that the photo really shows a recognizable dish.

### Why HITL uses LangGraph's `interrupt()`, not a bare `input()`

`hitl_gate_node` calls LangGraph's native `interrupt()` before any persisted file write -- saving
the meal to `plate_coach_history.json`, or exporting the conversation summary -- which pauses
the *graph itself*. The CLI reads the pending payload off `graph.get_state(config)`, prompts
`[Y/N]`, and resumes via `Command(resume=bool)`; the write happens inside the node, only on a
yes. This makes the confirmation a real gate the graph can't bypass, rather than a side-channel
prompt that a future change to the CLI could skip. It needs a checkpointer (`InMemorySaver`),
and the CLI starts a fresh thread per run so one analysis never inherits the last one's state.

### Every LLM call is metered

All LLM calls go through `llm.call_llm`, which appends `{node, model, prompt_tokens,
completion_tokens}` to the state's `usage_log`. The CLI prints a per-node cost table after each
task and `evals/cost_eval.py` uses the same log. Prices come from OpenRouter's public model
list (`pricing.py`, cached for a day). A free model costs $0, and its *theoretical* cost uses
its paid twin's price. When a `:free` model answers 429, `call_llm` falls back to the next free
model in `FREE_FALLBACK_MODELS`, and the log records which model actually answered.

## Guardrails: rewrite, don't just block

`agent/guardrails.py` doesn't reject responses that use binding/diagnostic phrasing ("you must
avoid X", "you have a deficiency in Y") — it rewrites them into softer, informational wording
("you may want to consider limiting X") and appends the fixed disclaimer. Refusal messages
(out-of-scope replies) skip the disclaimer, since they contain no dietary content to disclaim.
The evaluator node (`evaluator_node`) checks for fewer than 2 suggestions, no scorable
suggestion, leftover prescriptive language and numbers the LLM wrote itself, before the output
is considered final, with one bounded retry back to the planner rather than an outright block.

## Where the code lives, mapped to the concepts above

- `agent/vision_model.py` — runs the Part א׳ classifier, exported to ONNX (`agent/models/food_classifier.onnx`)
  with `onnxruntime`, and reproduces EfficientNet-B0's preprocessing in NumPy, so the agent
  needs no torch at runtime (predictions match the PyTorch checkpoint to ~1e-7).
- `agent/tools.py` — RAG lookup, profile aggregation, and the two Code-Exec functions
  (`compute_healthy_score`, `compute_delta`).
- `agent/nodes.py` — all 10 graph nodes plus the scope guard and the JSON-parsing helpers
  that tolerate real-world LLM output quirks.
- `agent/graph.py` — the three compiled LangGraph workflows and their conditional edges.
- `agent/guardrails.py` — prescriptive-language rewriting and the disclaimer.
- `agent/llm.py` — the OpenRouter `ChatOpenAI` client, including a `truststore`-based HTTP
  client so the agent works on networks with a TLS-intercepting proxy (see `RUNNING_AGENT.md`
  for the underlying SSL issue this solves).
- `agent/pricing.py` — placeholder per-model token pricing for the CLI's cost display.
- `agent/cli.py` — the interactive command loop, streaming, and the what-if/history/export
  commands.
