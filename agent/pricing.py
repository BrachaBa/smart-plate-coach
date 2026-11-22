"""Per-model token pricing for the CLI's cost summary and evals/cost_eval.py.

Prices come from OpenRouter's public model list (GET /api/v1/models, no key
needed), fetched once per process and cached to .pricing_cache.json for a day.
If the network is unavailable, the last cache or the built-in snapshot below is
used, so a cost is always shown.

Free models (":free") really cost $0. Their *theoretical* cost is priced at the
same model's paid variant (the slug without ":free") when OpenRouter lists one,
otherwise at REFERENCE_MODEL -- so a free run still shows what it would have cost.
"""

import json
import time
from pathlib import Path

import httpx

from agent.llm import DEFAULT_MODEL, OPENROUTER_BASE_URL, _build_http_client

CACHE_PATH = Path(".pricing_cache.json")
CACHE_TTL_S = 24 * 3600
REFERENCE_MODEL = "openai/gpt-4o-mini"

# USD per token, snapshot of OpenRouter's list (2026-11-22). Fallback only.
SNAPSHOT = {
    "google/gemma-4-26b-a4b-it:free": {"prompt": 0.0, "completion": 0.0},
    "google/gemma-4-26b-a4b-it": {"prompt": 0.0000000675, "completion": 0.000000225},
    "google/gemma-4-31b-it:free": {"prompt": 0.0, "completion": 0.0},
    "openai/gpt-4o-mini": {"prompt": 0.00000015, "completion": 0.0000006},
    "anthropic/claude-haiku-5.5": {"prompt": 0.0000001, "completion": 0.0000005},
    "anthropic/claude-sonnet-5.5": {"prompt": 0.000002, "completion": 0.00001},
}

_prices: dict | None = None


def _fetch_prices() -> dict:
    with _build_http_client() as client:
        response = client.get(f"{OPENROUTER_BASE_URL}/models", timeout=15)
        response.raise_for_status()
    prices = {}
    for model in response.json().get("data", []):
        pricing = model.get("pricing") or {}
        try:
            prices[model["id"]] = {"prompt": float(pricing["prompt"]), "completion": float(pricing["completion"])}
        except (KeyError, TypeError, ValueError):
            continue
    return prices


def get_prices() -> dict:
    """{model_id: {"prompt": usd_per_token, "completion": usd_per_token}}"""
    global _prices
    if _prices is not None:
        return _prices

    cached = None
    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
    if cached and time.time() - cached.get("fetched_at", 0) < CACHE_TTL_S:
        _prices = cached["prices"]
        return _prices

    try:
        _prices = _fetch_prices()
        try:
            CACHE_PATH.write_text(json.dumps({"fetched_at": time.time(), "prices": _prices}), encoding="utf-8")
        except OSError:
            pass
    except (httpx.HTTPError, ValueError):
        _prices = (cached or {}).get("prices") or dict(SNAPSHOT)
    return _prices


def model_rates(model: str) -> tuple[dict, dict, str]:
    """Return (actual_rate, theoretical_rate, theoretical_basis) for a model."""
    prices = get_prices()
    actual = prices.get(model) or SNAPSHOT.get(model) or {"prompt": 0.0, "completion": 0.0}
    if model.endswith(":free"):
        actual = {"prompt": 0.0, "completion": 0.0}
        paid_twin = model[: -len(":free")]
        if paid_twin in prices and (prices[paid_twin]["prompt"] or prices[paid_twin]["completion"]):
            return actual, prices[paid_twin], paid_twin
        reference = prices.get(REFERENCE_MODEL) or SNAPSHOT[REFERENCE_MODEL]
        return actual, reference, REFERENCE_MODEL
    return actual, actual, model


def estimate_cost(token_usage: dict, model: str = DEFAULT_MODEL) -> dict:
    """Cost of one usage record: {"actual_cost_usd", "theoretical_cost_usd", "theoretical_basis"}."""
    actual, theoretical, basis = model_rates(model)
    p = token_usage.get("prompt_tokens", 0)
    c = token_usage.get("completion_tokens", 0)
    return {
        "actual_cost_usd": round(p * actual["prompt"] + c * actual["completion"], 8),
        "theoretical_cost_usd": round(p * theoretical["prompt"] + c * theoretical["completion"], 8),
        "theoretical_basis": basis,
    }


def cost_breakdown(usage_log: list) -> dict:
    """Per-step and total cost of a run's usage_log.

    Returns {"steps": [{node, model, prompt_tokens, completion_tokens, actual_cost_usd,
    theoretical_cost_usd}], "total": {...same keys summed...}}.
    """
    steps = []
    for entry in usage_log or []:
        cost = estimate_cost(entry, entry.get("model") or DEFAULT_MODEL)
        steps.append({**entry, **cost})
    total = {
        "llm_calls": len(steps),
        "prompt_tokens": sum(s["prompt_tokens"] for s in steps),
        "completion_tokens": sum(s["completion_tokens"] for s in steps),
        "actual_cost_usd": round(sum(s["actual_cost_usd"] for s in steps), 8),
        "theoretical_cost_usd": round(sum(s["theoretical_cost_usd"] for s in steps), 8),
    }
    return {"steps": steps, "total": total}
