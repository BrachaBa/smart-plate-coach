"""Shared helpers for the three evals: the task set, running a task on a model,
and writing CSVs to evals/results/.

The evals measure the LLM workflow, so the vision step is pinned: each meal task
fixes the classifier's output (dish + confidence) instead of reading a photo.
That way every model gets byte-identical input and differences come only from
the LLMs. The classifier itself is evaluated in Part A (vision/results/,
82.8% test accuracy).

Run any eval from the project root, e.g.  python -m evals.scope_eval
"""

import csv
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Evals should not need a LangSmith key; tracing stays on only if one is set.
os.environ.setdefault("LANGSMITH_PROJECT", "plate-coach-evals")

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

import agent.nodes as nodes  # noqa: E402
from agent.graph import build_analysis_graph, build_question_graph, build_simulation_graph  # noqa: E402
from agent.pricing import cost_breakdown  # noqa: E402
from agent.tools import compute_delta, lookup_nutrition  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Common tasks: three photo analyses, two nutrition questions, one what-if.
TASKS = [
    {"id": "analyze_pizza", "kind": "analysis", "food": "pizza", "goal": "weight_loss"},
    {"id": "analyze_sushi", "kind": "analysis", "food": "sushi", "goal": "muscle_gain"},
    {"id": "analyze_caesar_salad", "kind": "analysis", "food": "caesar_salad", "goal": "balanced"},
    {"id": "qa_protein_compare", "kind": "question", "query": "How much protein is in sushi compared to steak?"},
    {"id": "qa_satfat_fries", "kind": "question", "query": "Are french fries high in saturated fat compared to grilled salmon?"},
    {"id": "whatif_pizza_salad", "kind": "whatif", "food": "pizza", "query": "What if I swap the pizza for a greek salad?"},
]


@contextmanager
def pinned_vision(food: str, confidence: float = 0.9):
    original = nodes.classify_image
    nodes.classify_image = lambda path, top_k=3: {
        "foods": [{"name": food, "confidence": confidence}],
        "portion_size": "medium",
    }
    try:
        yield
    finally:
        nodes.classify_image = original


def output_text(task: dict, state: dict) -> str:
    """A readable rendering of the agent's answer, as the judge and the CSV see it."""
    out = state.get("final_output") or {}
    if task["kind"] == "analysis":
        lines = [
            f"Identified: {out.get('food')}",
            f"Healthy Score: {(out.get('healthy_score') or {}).get('score')}/100",
            f"Profile: {out.get('qualitative_profile')}",
            "Suggestions:",
        ]
        for s in out.get("suggestions", []):
            lines.append(f"- {s['text']} (projected: {s.get('projected_score')})")
        return "\n".join(lines)
    if task["kind"] == "whatif":
        if out.get("refused"):
            return out.get("message", "")
        d = state.get("simulation_result")
        if not d:
            return f"ERROR: {state.get('error')}"
        return (f"Parsed change: {state.get('simulation_change')}\n"
                f"Before: score {d['before']['score']}, {d['before']['calories']} kcal\n"
                f"After: score {d['after']['score']}, {d['after']['calories']} kcal")
    return out.get("message", "")


def reference_data(task: dict, state: dict) -> dict:
    """What the judge compares against: database records + code-computed numbers."""
    if task["kind"] == "question":
        foods = state.get("retrieved_foods") or []
        records = [{k: v for k, v in r.items() if k not in ("query", "is_default", "aliases")}
                   for r in lookup_nutrition(foods)["matched"] if not r.get("is_default")]
        return {"nutrition_records_per_100g": records}
    meal = lookup_nutrition([task["food"]])
    ref = {"nutrition_record_per_100g": {k: v for k, v in meal["matched"][0].items() if k not in ("query", "aliases")}}
    if task["kind"] == "analysis":
        ref["healthy_score_computed_by_code"] = state.get("healthy_score")
        ref["goal"] = task["goal"]
    else:
        ref["expected_change"] = "swap pizza -> greek_salad"
        ref["expected_result_computed_by_code"] = compute_delta(
            meal["matched"], {"action": "swap", "food": "pizza", "replacement": "greek salad"})["after"]
    return ref


def run_task(task: dict, model: str, strict: bool = True, attempts: int = 2) -> dict:
    """Run one task end to end on one model. Never raises: errors become status=error.

    A dropped connection gets one more attempt (it says nothing about the model);
    a rate limit or a bad answer does not, since those are the model's reliability.
    """
    for attempt in range(attempts):
        result = _run_task_once(task, model, strict)
        if "ConnectionError" not in result["error"] or attempt == attempts - 1:
            return result
        time.sleep(10)
    return result


class UsageRecorder(BaseCallbackHandler):
    """Records every LLM call's tokens as it finishes, outside the graph state.

    A run that fails part-way returns no state, so its usage_log is lost -- but
    the calls made before the failure were still billed. This keeps them.
    """

    def __init__(self):
        self.started: dict = {}
        self.log: list = []

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, invocation_params=None, **kwargs):
        params = invocation_params or kwargs.get("invocation_params") or {}
        self.started[run_id] = {
            "node": (metadata or {}).get("langgraph_node", "?"),
            "model": params.get("model") or params.get("model_name") or "?",
        }

    def on_llm_end(self, response, *, run_id, **kwargs):
        info = self.started.pop(run_id, {"node": "?", "model": "?"})
        usage = {}
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            pass
        self.log.append({
            **info,
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        })


def _run_task_once(task: dict, model: str, strict: bool) -> dict:
    started = time.perf_counter()
    state, status, error = {}, "ok", ""
    recorder = UsageRecorder()
    callbacks = {"callbacks": [recorder]}
    try:
        if task["kind"] == "analysis":
            with pinned_vision(task["food"]):
                graph = build_analysis_graph()
                state = graph.invoke(
                    {"image_path": f"{task['food']}.jpg", "goal": task["goal"], "model": model, "strict_model": strict},
                    {"configurable": {"thread_id": f"{task['id']}-{model}-{time.time()}"}, **callbacks},
                )
        elif task["kind"] == "question":
            state = build_question_graph().invoke(
                {"user_query": task["query"], "model": model, "strict_model": strict}, callbacks
            )
        else:
            meal = lookup_nutrition([task["food"]])
            state = build_simulation_graph().invoke({
                "user_query": task["query"], "model": model, "strict_model": strict,
                "nutrition_lookup": meal, "vision_result": {"portion_size": "medium"},
            }, callbacks)
            if state.get("error"):
                status, error = "parse_error", state["error"]
    except Exception as exc:  # recorded, not raised: a failing model is a data point
        status, error = "error", f"{type(exc).__name__}: {str(exc)[:200]}"
    latency = time.perf_counter() - started
    # A failed run has no state; price the calls the recorder saw instead, and
    # say the figure covers only the calls made before the failure.
    usage_log = state.get("usage_log") or recorder.log
    costs = cost_breakdown(usage_log)
    cost_note = "" if state.get("usage_log") or not recorder.log else "partial: calls before the failure"
    return {
        "cost_note": cost_note,
        "task": task,
        "model": model,
        "status": status,
        "error": error,
        "latency_s": round(latency, 2),
        "state": state,
        "text": output_text(task, state) if status != "error" else "",
        "costs": costs,
    }


def write_csv(name: str, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: opens cleanly in Excel
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def progress(msg: str):
    print(msg, file=sys.stderr, flush=True)
