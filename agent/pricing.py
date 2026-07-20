"""Placeholder per-model token pricing, used for the CLI's cost display and
evals/cost_eval.py's theoretical-cost calculation for free models.

Rates are USD per 1000 tokens. These are illustrative placeholders -- swap in
real OpenRouter pricing for the models you actually configure in .env before
treating the numbers as accurate.
"""

from agent.llm import DEFAULT_MODEL, PAID_MODEL

# {model_name: {"prompt": $/1K prompt tokens, "completion": $/1K completion tokens}}
PRICING_TABLE = {
    DEFAULT_MODEL: {"prompt": 0.0, "completion": 0.0},  # free-tier model
    PAID_MODEL: {"prompt": 0.003, "completion": 0.015},  # placeholder paid-model rate
}

# Used when a free model's *theoretical* cost is wanted for comparison (spec 3.6 /
# evals/cost_eval.py): "what would this have cost at an equivalent paid rate?"
THEORETICAL_RATE = {"prompt": 0.003, "completion": 0.015}


def estimate_cost(token_usage: dict, model: str = DEFAULT_MODEL) -> dict:
    """Returns {"actual_cost_usd": float, "theoretical_cost_usd": float}.

    actual_cost_usd uses the model's real configured rate (0 for free models).
    theoretical_cost_usd always uses THEORETICAL_RATE, so free-model runs still
    show a meaningful cost figure for comparison purposes.
    """
    rate = PRICING_TABLE.get(model, {"prompt": 0.0, "completion": 0.0})
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)

    actual = (prompt_tokens / 1000) * rate["prompt"] + (completion_tokens / 1000) * rate["completion"]
    theoretical = (
        (prompt_tokens / 1000) * THEORETICAL_RATE["prompt"]
        + (completion_tokens / 1000) * THEORETICAL_RATE["completion"]
    )

    return {"actual_cost_usd": round(actual, 6), "theoretical_cost_usd": round(theoretical, 6)}
