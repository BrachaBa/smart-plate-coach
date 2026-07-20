"""Shared LangGraph state schema for the Plate Coach agent.

A plain TypedDict rather than a Pydantic model: every field here is produced by our
own trusted node code, not external untrusted input requiring schema validation --
the actually untrusted input (image path, free-text questions) is validated
explicitly inside the gate/scope nodes instead. TypedDict also keeps `.invoke()` /
`.stream()` return values plain dicts, which is the simplest shape for the CLI to
print and for LangGraph's `stream_mode="updates"` deltas to merge.
"""

from typing import TypedDict


class PlateCoachState(TypedDict, total=False):
    # --- session / input ---
    image_path: str
    goal: str  # "weight_loss" | "maintain" | "balanced" | "muscle_gain"
    user_query: str  # raw free-text input for scope_guard / simulation routing

    # --- scope guard ---
    in_scope: bool
    refusal_message: str | None

    # --- vision intake + gate ---
    vision_result: dict  # {"foods": [{"name": str, "confidence": float}, ...], "portion_size": str}
    gate_ok: bool
    gate_reason: str | None

    # --- nutrition analysis (RAG) ---
    nutrition_lookup: dict  # {"matched": [...], "unmatched": [...]}
    nutrition_profile: dict  # aggregated macro/veg profile, see agent.tools.aggregate_profile
    qualitative_profile: str  # LLM-phrased qualitative summary, e.g. "protein: good, sugar: high"

    # --- scoring (Code-Exec) ---
    healthy_score: dict  # {"score": int, "breakdown": {...}}

    # --- improvement planner ---
    suggestions: list  # [{"text": str, "projected_score": int}, ...]

    # --- simulation ("what if") ---
    simulation_query: str | None
    simulation_result: dict | None  # {"before": {...}, "after": {...}, "delta": {...}}

    # --- goal adjustment ---
    adjusted_suggestions: list

    # --- meal history ---
    history_photos: list
    history_summary: str | None

    # --- evaluator ---
    evaluator_ok: bool
    evaluator_notes: str | None
    retry_count: int

    # --- HITL ---
    pending_write: dict | None  # {"target": str, "payload": ...} awaiting [Y/N] confirmation

    # --- output / bookkeeping ---
    final_output: dict
    token_usage: dict  # running {"prompt_tokens": int, "completion_tokens": int}
    error: str | None
