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

`agent/data/nutrition_facts.json` is a small (~50 food) local table of calories/macros/veg
content per food, looked up by `agent/tools.py::lookup_nutrition`. This is deliberately a plain
JSON file with keyword/alias lookup, not a vector database — the spec calls this pattern RAG
because the LLM's job (turning matched records into readable qualitative language, e.g.
"protein: good, sugar: high") is grounded in retrieved facts rather than invented ones, and at
~50 records a similarity-search index buys nothing a dict lookup doesn't already give.
Foods the vision model recognizes but that aren't in the table get a neutral default profile
and are flagged as unmatched, rather than the LLM guessing plausible-looking numbers for them.

## The Healthy Score formula

`agent/tools.py::compute_healthy_score` is a weighted 0-100 score: protein (0.25), vegetable
content (0.20), a carb-excess penalty (0.15), a sugar penalty (0.20), and a saturated-fat
penalty (0.20), each sub-score clipped to [0, 100] before weighting so no single extreme macro
(e.g. a very high-protein meal) can push the total out of range on its own. The thresholds
(`PROTEIN_TARGET_G`, `SUGAR_LIMIT_G`, `SATFAT_LIMIT_G`, `CARB_LIMIT_G`) are general per-meal
reference figures, not a personalized medical target — the agent never tailors these to an
individual's body data, in line with the spec's ethical constraints (Part ד׳).

## The LangGraph workflow

`agent/graph.py` compiles three separate graphs from the same node module
(`agent/nodes.py`), because simulation and history analysis don't belong on the main linear
path:

- **`build_analysis_graph()`** — the core flow: `scope_guard → vision_intake → input_gate →
  nutrition_analysis → scoring → improvement_planner → goal_adjuster → evaluator → hitl_gate`.
  The scope guard and input gate can short-circuit to `END` (refusal, or "please provide a
  clearer photo"); the evaluator can loop back to the planner once if something's inconsistent
  before it gives up and returns what it has.
- **`build_simulation_graph()`** — a single node wrapping free-form "what if" questions,
  invoked repeatedly against the last analysis's data without re-running vision/RAG/scoring
  from scratch each time.
- **`build_history_graph()`** — a single node that runs vision+RAG+scoring over up to 20 stored
  photos and asks the LLM only for the trend *summary*, not the trend *numbers* (those are
  computed the same deterministic way as everything else).

### Why a scope guard in front of node 1

The spec requires the agent to actively refuse out-of-domain requests ("write me code", "what's
the weather"), not just fail to answer them well. `scope_guard_node` asks a fast LLM call
whether free-text input is in-scope *before* any vision/nutrition processing happens, so an
off-topic question never reaches the parts of the pipeline that assume a meal-analysis context.
An image path is always treated as in-scope, since submitting a photo is the core task itself.

### Why HITL uses LangGraph's `interrupt()`, not a bare `input()`

`hitl_gate_node` calls LangGraph's native `interrupt()` before any persisted file write
(meal history, profile), which pauses the *graph itself* — the CLI reads the pending payload
off `graph.get_state(config)`, prompts `[Y/N]`, and resumes via `Command(resume=bool)`. This
matters because it makes the confirmation step a real gate the graph can't bypass, rather than
a side-channel prompt that a future change to the CLI could accidentally skip. It requires the
graph to be compiled with a checkpointer (`InMemorySaver` — sufficient since one CLI process is
one session) so LangGraph has somewhere to persist state across the pause.

## Guardrails: rewrite, don't just block

`agent/guardrails.py` doesn't reject responses that use binding/diagnostic phrasing ("you must
avoid X", "you have a deficiency in Y") — it rewrites them into softer, informational wording
("you may want to consider limiting X") and appends the fixed disclaimer. Refusal messages
(out-of-scope replies) skip the disclaimer, since they contain no dietary content to disclaim.
The evaluator node (`evaluator_node`) checks for leftover prescriptive language and missing
suggestions before the output is considered final, with one bounded retry back to the planner
rather than an outright block.

## Where the code lives, mapped to the concepts above

- `agent/vision_model.py` — loads the Part א׳ checkpoint (`vision/checkpoints/config4_full.pt`)
  and classifies a photo; reuses `vision/vision_model.py`'s architecture/transforms and
  `vision/train.py`'s device detection as-is.
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
