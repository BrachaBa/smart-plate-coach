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
import json
import statistics
from pathlib import Path

from agent.llm import DEFAULT_MODEL, JUDGE_MODEL, PAID_MODEL, call_llm
from agent.nodes import _parse_json_response
from evals.common import TASKS, progress, reference_data, run_task, write_csv

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
    description = task.get("query") or f"Analyze a meal photo of {task['food']} for a user whose goal is {task.get('goal')}."
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


def run(models: list[str], repeats: int) -> None:
    if JUDGE_MODEL in models:
        raise SystemExit(f"The judge ({JUDGE_MODEL}) must not be one of the compared models.")
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

    summary = []
    for model in models:
        mine = [r for r in rows if r["model"] == model]
        ok = [r for r in mine if r["status"] == "ok"]
        judged = [r for r in ok if "overall" in r]
        entry = {
            "model": model,
            "runs": len(mine),
            "success_rate_pct": round(100 * len(ok) / len(mine), 1) if mine else 0,
            "avg_latency_s": round(statistics.mean(r["latency_s"] for r in ok), 2) if ok else "",
        }
        for c in CRITERIA + ["overall"]:
            entry[f"avg_{c}"] = round(statistics.mean(r[c] for r in judged), 2) if judged else ""
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
    args = parser.parse_args()
    run([m.strip() for m in args.models.split(",") if m.strip()], args.repeats)
