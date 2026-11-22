"""Eval 1 -- model comparison: the same tasks on different models.

For every (model, task) pair: wall-clock time (measured automatically), success
or failure (reliability), and five 1-5 quality scores from an LLM-as-judge that
follows evals/rubric.md. The judge is one fixed model (JUDGE_MODEL) for every
comparison, and it is not one of the models under test.

    python -m evals.compare_models
    python -m evals.compare_models --models a,b,c --repeats 2

Writes evals/results/compare_models.csv (one row per run) and
evals/results/compare_models_summary.csv (per-model averages).
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from agent.llm import DEFAULT_MODEL, JUDGE_MODEL, PAID_MODEL, call_llm
from agent.nodes import _parse_json_response
from evals.common import RESULTS_DIR, TASKS, progress, reference_data, run_task, write_csv

# What a correct output looks like for each task kind, so the judge grades it
# against the agent's actual contract (e.g. a what-if is a structured,
# code-computed before/after, not prose).
EXPECTED_SHAPE = {
    "analysis": (
        "Analyze a meal photo of {food} for a user whose goal is {goal}. Expected output: the "
        "dish, a Healthy Score computed by code, a short qualitative nutrition profile, and "
        "2-4 short, soft, number-free suggestions, each with a projected score computed by code."
    ),
    "question": (
        "Answer the user's nutrition question: \"{query}\". Expected output: a direct 2-5 "
        "sentence answer grounded in the reference records."
    ),
    "whatif": (
        "What-if simulation on a meal of {food}. User asked: \"{query}\". Expected output, by "
        "design a structured result rather than prose: the change the LLM parsed from the "
        "question (action/food/replacement), and the before/after Healthy Score and calories "
        "computed by code. Grade the parse and whether the numbers match the reference."
    ),
}

RUBRIC = (Path(__file__).resolve().parent / "rubric.md").read_text(encoding="utf-8")
CRITERIA = ["relevance", "factual_accuracy", "completeness", "format", "data_reliability"]

JUDGE_PROMPT = """You are a strict, impartial evaluator of a nutrition agent's output.
Grade it with this rubric:

{rubric}

TASK GIVEN TO THE AGENT:
{task}

REFERENCE DATA (ground truth from the agent's database and its own code):
{reference}

AGENT OUTPUT:
<output>
{output}
</output>

Return ONLY the JSON object described under "Judge output format", nothing else.
"""


def judge(task: dict, text: str, reference: dict) -> dict:
    description = EXPECTED_SHAPE[task["kind"]].format(**task)
    prompt = JUDGE_PROMPT.format(
        rubric=RUBRIC, task=description, reference=json.dumps(reference, indent=1), output=text or "(empty)"
    )
    raw, usage = call_llm({"model": JUDGE_MODEL, "strict_model": True}, "judge", prompt, temperature=0)
    scores = _parse_json_response(raw)
    result = {c: int(scores[c]) for c in CRITERIA}
    result["overall"] = round(statistics.mean(result.values()), 2)
    result["judge_comment"] = str(scores.get("comment", ""))[:300]
    result["judge_cost_usd"] = usage["usage_log"][0]
    return result


def _load_previous(models: list[str]) -> list[dict]:
    """Rows of an earlier run for models *not* in this run (for --merge)."""
    path = RESULTS_DIR / "compare_models.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["model"] not in models]


def _num(value):
    return float(value) if value not in (None, "") else None


def run(models: list[str], repeats: int, merge: bool = False) -> None:
    if JUDGE_MODEL in models:
        raise SystemExit(f"The judge ({JUDGE_MODEL}) must not be one of the compared models.")
    previous = _load_previous(models) if merge else []
    rows = []
    for model in models:
        for task in TASKS:
            for rep in range(1, repeats + 1):
                progress(f"[compare_models] {model} | {task['id']} | run {rep}")
                result = run_task(task, model)
                row = {
                    "model": model,
                    "task": task["id"],
                    "run": rep,
                    "status": result["status"],
                    "latency_s": result["latency_s"],
                    "llm_calls": result["costs"]["total"]["llm_calls"],
                    "prompt_tokens": result["costs"]["total"]["prompt_tokens"],
                    "completion_tokens": result["costs"]["total"]["completion_tokens"],
                    "cost_usd": f"{result['costs']['total']['actual_cost_usd']:.8f}",
                    "theoretical_cost_usd": f"{result['costs']['total']['theoretical_cost_usd']:.8f}",
                }
                if result["status"] == "ok":
                    try:
                        scores = judge(task, result["text"], reference_data(task, result["state"]))
                        scores.pop("judge_cost_usd", None)
                        row.update(scores)
                    except Exception as exc:
                        row["judge_comment"] = f"judge failed: {type(exc).__name__}: {exc}"[:300]
                row["error"] = result["error"]
                row["output"] = result["text"].replace("\n", " | ")[:600]
                rows.append(row)

    rows = previous + rows
    summary = []
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        ok = [r for r in mine if r["status"] == "ok"]
        judged = [r for r in ok if _num(r.get("overall")) is not None]
        entry = {
            "model": model,
            "runs": len(mine),
            "success_rate_pct": round(100 * len(ok) / len(mine), 1) if mine else 0,
            "avg_latency_s": round(statistics.mean(_num(r["latency_s"]) for r in ok), 2) if ok else "",
        }
        for c in CRITERIA + ["overall"]:
            entry[f"avg_{c}"] = round(statistics.mean(_num(r[c]) for r in judged), 2) if judged else ""
        entry["total_cost_usd"] = f"{sum(float(r['cost_usd']) for r in mine):.6f}"
        entry["total_theoretical_cost_usd"] = f"{sum(float(r['theoretical_cost_usd']) for r in mine):.6f}"
        entry["judge_model"] = JUDGE_MODEL
        summary.append(entry)

    print("wrote", write_csv("compare_models.csv", rows))
    print("wrote", write_csv("compare_models_summary.csv", summary))
    for s in summary:
        print(f"{s['model']}: success {s['success_rate_pct']}%, avg {s['avg_latency_s']} s, "
              f"overall {s['avg_overall']}/5, cost ${s['total_cost_usd']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=f"{DEFAULT_MODEL},google/gemma-4-26b-a4b-it:free,{PAID_MODEL},openai/gpt-4o-mini",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="keep the earlier results of models not in --models (e.g. re-run only some)")
    args = parser.parse_args()
    run([m.strip() for m in args.models.split(",") if m.strip()], args.repeats, args.merge)
