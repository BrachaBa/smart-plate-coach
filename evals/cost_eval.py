"""Eval 2 -- cost: what does each step of each common task cost, per model?

Runs the common tasks (evals.common.TASKS) on a free model and paid models and
prices every LLM call from its token usage x the model's OpenRouter price
(agent.pricing). Free models show the actual cost ($0) next to the theoretical
cost -- the same tokens priced at the model's paid variant (or a reference paid
model when there is none), so the comparison is meaningful even when nothing
was charged.

    python -m evals.cost_eval
    python -m evals.cost_eval --models a,b,c

Writes evals/results/cost_eval.csv (one row per LLM call, i.e. per step) and
evals/results/cost_eval_summary.csv (per task and model totals, plus a TOTAL row).
"""

import argparse

from agent.llm import DEFAULT_MODEL, PAID_MODEL
from evals.common import TASKS, progress, run_task, write_csv


def run(models: list[str]) -> None:
    step_rows, summary_rows = [], []
    for model in models:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "actual_cost_usd": 0.0, "theoretical_cost_usd": 0.0, "llm_calls": 0}
        for task in TASKS:
            progress(f"[cost_eval] {model} | {task['id']}")
            result = run_task(task, model)
            for i, step in enumerate(result["costs"]["steps"], 1):
                step_rows.append({
                    "model": model,
                    "task": task["id"],
                    "step": i,
                    "node": step["node"],
                    "answered_by": step["model"],
                    "prompt_tokens": step["prompt_tokens"],
                    "completion_tokens": step["completion_tokens"],
                    "actual_cost_usd": f"{step['actual_cost_usd']:.8f}",
                    "theoretical_cost_usd": f"{step['theoretical_cost_usd']:.8f}",
                    "theoretical_basis": step["theoretical_basis"],
                })
            t = result["costs"]["total"]
            summary_rows.append({
                "model": model,
                "task": task["id"],
                "status": result["status"],
                "latency_s": result["latency_s"],
                "llm_calls": t["llm_calls"],
                "prompt_tokens": t["prompt_tokens"],
                "completion_tokens": t["completion_tokens"],
                "actual_cost_usd": f"{t['actual_cost_usd']:.8f}",
                "theoretical_cost_usd": f"{t['theoretical_cost_usd']:.8f}",
                "error": result["error"],
            })
            for k in totals:
                totals[k] += t[k]
        summary_rows.append({
            "model": model,
            "task": "TOTAL",
            "status": "",
            "latency_s": "",
            "llm_calls": totals["llm_calls"],
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "actual_cost_usd": f"{totals['actual_cost_usd']:.8f}",
            "theoretical_cost_usd": f"{totals['theoretical_cost_usd']:.8f}",
            "error": "",
        })
        print(f"{model}: {totals['llm_calls']} calls, actual ${totals['actual_cost_usd']:.6f}, "
              f"theoretical ${totals['theoretical_cost_usd']:.6f}")

    print("wrote", write_csv("cost_eval.csv", step_rows))
    print("wrote", write_csv("cost_eval_summary.csv", summary_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=f"{DEFAULT_MODEL},{PAID_MODEL},openai/gpt-4o-mini")
    run([m.strip() for m in parser.parse_args().models.split(",") if m.strip()])
