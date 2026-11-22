"""Shared LangGraph state schema for the Plate Coach agent.

A plain TypedDict rather than a Pydantic model: every field here is produced by our
own trusted node code, not external untrusted input requiring schema validation --
the actually untrusted input (image path, free-text questions) is validated
explicitly inside the gate/scope nodes instead. TypedDict also keeps `.invoke()` /
`.stream()` return values plain dicts, which is the simplest shape for the CLI to
print and for the evals to read.
"""

from typing import TypedDict


class PlateCoachState(TypedDict, total=False):
    # --- session / input ---
    image_path: str
    goal: str  # "weight_loss" | "maintain" | "balanced" | "muscle_gain"
    user_query: str  # free-text question / what-if, checked by scope_guard
    model: str | None  # OpenRouter model for this run (None -> DEFAULT_MODEL); evals vary it
    strict_model: bool  # evals: no free-model fallback, so the named model really answered
    save_history: bool  # offer to save the analyzed meal (HITL) at the end of the run

    # --- scope guard ---
    in_scope: bool
    refusal_reason: str | None  # "out_of_scope" | "prompt_injection"

    # --- vision intake + gate ---
    vision_result: dict  # {"foods": [{"name": str, "confidence": float}, ...], "portion_size": str}
    gate_ok: bool
    gate_reason: str | None

    # --- nutrition analysis (RAG) ---
    nutrition_lookup: dict  # {"matched": [...], "unmatched": [...]}
    nutrition_profile: dict  # aggregated macro/veg profile, see agent.tools.aggregate_profile
    qualitative_profile: str  # LLM-phrased qualitative summary, e.g. "protein: good, sugar: high"
    retrieved_foods: list  # records the nutrition_qa node grounded its answer in

    # --- scoring (Code-Exec) ---
    healthy_score: dict  # {"score": int, "breakdown": {...}}

    # --- improvement planner ---
    suggestions: list  # [{"text": str, "projected_score": int | None, "change": {...}}, ...]

    # --- simulation ("what if") ---
    simulation_result: dict | None  # {"before": {...}, "after": {...}, "delta": {...}}
    simulation_change: dict | None

    # --- goal adjustment ---
    adjusted_suggestions: list

    # --- meal history ---
    history_photos: list
    history_summary: str | None
    history_meals: list
    history_stats: dict
    history_skipped: list

    # --- evaluator ---
    evaluator_ok: bool
    evaluator_notes: str | None
    retry_count: int

    # --- HITL ---
    pending_write: dict | None  # {"target", "mode", "content", "description"} awaiting [Y/N]
    write_result: str | None

    # --- output / bookkeeping ---
    final_output: dict
    usage_log: list  # one {"node", "model", "prompt_tokens", "completion_tokens"} per LLM call
    token_usage: dict  # running {"prompt_tokens": int, "completion_tokens": int}
    error: str | None
