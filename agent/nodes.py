"""LangGraph node functions for the Plate Coach agent.

Ten nodes total (spec section 3.2), plus a scope guard that sits in front of
node 1: scope_guard, vision_intake, input_gate, nutrition_analysis, scoring,
improvement_planner, simulation_handler, goal_adjuster, history_analyzer,
evaluator, hitl_gate.
"""

import json
import re

from langgraph.types import interrupt

from agent.guardrails import apply_guardrails, find_prescriptive_language
from agent.llm import get_chat_model, extract_usage
from agent.state import PlateCoachState
from agent.tools import aggregate_profile, compute_delta, compute_healthy_score, lookup_nutrition
from agent.vision_model import classify_image

GATE_CONFIDENCE_THRESHOLD = 0.35
SCOPE_STATEMENT = (
    "I'm Plate Coach -- I only analyze meal photos for nutrition, healthy scoring, "
    "what-if simulations, and general dietary suggestions. I can't help with that request."
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json_response(content: str):
    """Parse an LLM's JSON response, tolerating markdown code fences (```json ... ```)
    that models commonly wrap structured output in even when told not to."""
    return json.loads(_CODE_FENCE_RE.sub("", content).strip())


SCOPE_CHECK_PROMPT = """You are a scope router for Plate Coach, a nutrition-analysis agent.
Plate Coach's ONLY job: analyzing meal photos, computing healthy scores, running
what-if diet simulations, and giving general (non-medical) dietary suggestions.

Decide if the following user input is IN scope (an image-analysis request, a
nutrition/diet question about a meal, or a what-if question about a meal) or
OUT of scope (anything else: code, weather, general trivia, investment advice,
medical diagnosis requests, etc).

User input: {user_query}

Respond with exactly one word: INSCOPE or OUTOFSCOPE.
"""


def scope_guard_node(state: PlateCoachState) -> dict:
    """Front-door node: refuses out-of-domain free-text before any vision/nutrition
    processing happens. An image path is always in-scope (it's the core task input)."""
    query = state.get("user_query", "")
    if not query:
        return {"in_scope": True}

    model = get_chat_model()
    response = model.invoke(SCOPE_CHECK_PROMPT.format(user_query=query))
    usage = extract_usage(response)
    token_usage = dict(state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0})
    token_usage["prompt_tokens"] += usage["prompt_tokens"]
    token_usage["completion_tokens"] += usage["completion_tokens"]

    in_scope = "INSCOPE" in response.content.upper()
    result = {"in_scope": in_scope, "token_usage": token_usage}
    if not in_scope:
        result["refusal_message"] = SCOPE_STATEMENT
        result["final_output"] = {"refused": True, "message": SCOPE_STATEMENT}
    return result


def vision_intake_node(state: PlateCoachState) -> dict:
    """Node 1: classify the meal photo via the Part א' checkpoint."""
    result = classify_image(state["image_path"], top_k=3)
    return {"vision_result": result}


def input_gate_node(state: PlateCoachState) -> dict:
    """Node 2 (Gate/Guardrail): validate the vision result before nutrition analysis.

    Loops back to vision_intake (via a conditional edge) if confidence is too low --
    the CLI is responsible for prompting the user for a new/clearer image in that case.
    """
    vision_result = state.get("vision_result")
    if not vision_result or not vision_result.get("foods"):
        return {"gate_ok": False, "gate_reason": "No food identified in the image."}

    top_confidence = vision_result["foods"][0]["confidence"]
    if top_confidence < GATE_CONFIDENCE_THRESHOLD:
        return {
            "gate_ok": False,
            "gate_reason": (
                f"Low identification confidence ({top_confidence:.0%}). "
                "Please provide a clearer or closer photo of the meal."
            ),
        }
    return {"gate_ok": True, "gate_reason": None}


NUTRITION_PROFILE_PROMPT = """You are the Plate Coach nutrition-analysis component.
Given this looked-up nutrition data for a meal's foods, write a short qualitative
nutrition profile (2-3 short phrases, e.g. "protein: good, carbs: moderate,
veg: low, sugar: high"). Do not invent numbers -- only describe qualitatively.

Nutrition data (from a trusted database, already aggregated for this meal):
{profile_json}

Respond with only the qualitative profile line, nothing else.
"""


def nutrition_analysis_node(state: PlateCoachState) -> dict:
    """Node 3: RAG lookup of macro/veg data, then an LLM call to phrase it qualitatively.

    The LLM never computes numbers here -- lookup_nutrition/aggregate_profile (Code-Exec
    over the RAG database) already produced the real numeric profile; the LLM only turns
    it into readable qualitative language.
    """
    vision_result = state["vision_result"]
    food_names = [f["name"] for f in vision_result["foods"][:1]]  # top prediction only

    lookup = lookup_nutrition(food_names)
    profile = aggregate_profile(lookup["matched"], portion_size=vision_result.get("portion_size", "medium"))

    model = get_chat_model()
    prompt = NUTRITION_PROFILE_PROMPT.format(profile_json=json.dumps(profile, indent=2))
    response = model.invoke(prompt)
    usage = extract_usage(response)

    token_usage = dict(state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0})
    token_usage["prompt_tokens"] += usage["prompt_tokens"]
    token_usage["completion_tokens"] += usage["completion_tokens"]

    return {
        "nutrition_lookup": lookup,
        "nutrition_profile": profile,
        "token_usage": token_usage,
        "qualitative_profile": response.content,
    }


def scoring_node(state: PlateCoachState) -> dict:
    """Node 4: deterministic Healthy Score via Code-Exec (agent.tools), not LLM arithmetic."""
    score = compute_healthy_score(state["nutrition_profile"])
    return {"healthy_score": score}


PLANNER_PROMPT = """You are the Plate Coach improvement planner.
The user's meal has this nutrition profile and Healthy Score:
{profile_json}
Current Healthy Score: {score}/100

Propose 2 to 4 small, concrete, actionable changes to this meal (e.g. swapping
soda for water, adding a side salad, removing a high-sugar item). Respond with
ONLY a JSON array, no other text, where each element has this exact shape:
{{"text": "<human-readable suggestion, e.g. 'Swap soda for water'>",
  "action": "remove"|"add"|"swap",
  "food": "<food name being removed/swapped-out>",
  "replacement": "<food name being added, or null if action is 'remove'>"}}

Do not include any calorie/score numbers in the text -- those are computed
separately by the system, not by you.
"""


def improvement_planner_node(state: PlateCoachState) -> dict:
    """Node 5: LLM proposes 2-4 structured suggestions; each is re-scored via
    Code-Exec (compute_delta) using the LLM-chosen action/food/replacement,
    never by the LLM guessing the resulting number itself."""
    profile = state["nutrition_profile"]
    score = state["healthy_score"]["score"]

    model = get_chat_model()
    prompt = PLANNER_PROMPT.format(profile_json=json.dumps(profile, indent=2), score=score)
    response = model.invoke(prompt)
    usage = extract_usage(response)

    token_usage = dict(state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0})
    token_usage["prompt_tokens"] += usage["prompt_tokens"]
    token_usage["completion_tokens"] += usage["completion_tokens"]

    try:
        proposed = _parse_json_response(response.content)
    except json.JSONDecodeError:
        proposed = []

    meal_records = state.get("nutrition_lookup", {}).get("matched", [])
    meal_foods = {r["query"].lower() for r in meal_records}
    portion_size = state.get("vision_result", {}).get("portion_size", "medium")

    suggestions = []
    for item in proposed[:4]:
        text = item.get("text", "").strip()
        if not text:
            continue
        change = _normalize_change({
            "action": item.get("action"),
            "food": item.get("food"),
            "replacement": item.get("replacement"),
        })
        # Only trust the LLM's "remove"/"swap" target if that food was actually
        # part of this meal -- otherwise compute_delta would silently drop
        # nutrition data for something never eaten, producing a wrong score.
        removable_food = change.get("food")
        food_is_in_meal = not removable_food or removable_food.lower() in meal_foods
        projected_score = None
        if change.get("action") in ("remove", "swap") and not food_is_in_meal:
            projected_score = None
        else:
            try:
                delta = compute_delta(meal_records, change, portion_size)
                projected_score = delta["after"]["score"]
            except (KeyError, TypeError):
                projected_score = None
        suggestions.append({"text": text, "projected_score": projected_score})

    return {"suggestions": suggestions, "token_usage": token_usage}


SIMULATION_PARSE_PROMPT = """You are the Plate Coach what-if simulation parser.
The user asked: "{query}"

Identify the change they're describing as JSON with this exact shape:
{{"action": "remove"|"add"|"swap", "food": "<food name being removed, or null if action is 'add'>", "replacement": "<food name being added, or null if action is 'remove'>"}}

For "remove": only "food" is set, "replacement" is null.
For "add": only "replacement" is set (the food being added), "food" is null.
For "swap": both are set -- "food" is removed, "replacement" is added.

Only respond with the JSON object, nothing else.
"""


def _normalize_change(change: dict) -> dict:
    """Defensively normalize an LLM-parsed change: models sometimes put the
    food being added into "food" instead of "replacement" for an "add" action
    despite the prompt's schema. Treat that as the intended replacement rather
    than silently producing a zero-effect delta."""
    action = change.get("action")
    food = change.get("food")
    replacement = change.get("replacement")
    if action == "add" and food and not replacement:
        return {"action": "add", "food": None, "replacement": food}
    return change


def simulation_handler_node(state: PlateCoachState) -> dict:
    """Node 6: parses a free-form what-if question into a structured change,
    then recomputes the delta via Code-Exec (never LLM arithmetic)."""
    query = state["simulation_query"]
    model = get_chat_model()
    response = model.invoke(SIMULATION_PARSE_PROMPT.format(query=query))
    usage = extract_usage(response)

    token_usage = dict(state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0})
    token_usage["prompt_tokens"] += usage["prompt_tokens"]
    token_usage["completion_tokens"] += usage["completion_tokens"]

    try:
        change = _normalize_change(_parse_json_response(response.content))
    except json.JSONDecodeError:
        return {
            "simulation_result": None,
            "error": "Could not understand that what-if question -- try phrasing it as a specific swap, addition, or removal.",
            "token_usage": token_usage,
        }

    meal_records = state.get("nutrition_lookup", {}).get("matched", [])
    portion_size = state.get("vision_result", {}).get("portion_size", "medium")
    delta = compute_delta(meal_records, change, portion_size)
    return {"simulation_result": delta, "token_usage": token_usage, "error": None}


GOAL_PHRASING = {
    "weight_loss": "with an emphasis on calorie and portion reduction",
    "maintain": "with an emphasis on balance and consistency",
    "balanced": "with an emphasis on overall nutritional balance",
    "muscle_gain": "with an emphasis on protein intake and adequate calories",
}


def goal_adjuster_node(state: PlateCoachState) -> dict:
    """Node 7: reword suggestions to fit the session goal -- wording only,
    never touches the numeric projected_score computed by Code-Exec."""
    goal = state.get("goal", "balanced")
    emphasis = GOAL_PHRASING.get(goal, GOAL_PHRASING["balanced"])
    suggestions = state.get("suggestions", [])

    adjusted = [
        {**s, "text": f"{s['text']} ({emphasis})"} for s in suggestions
    ]
    return {"adjusted_suggestions": adjusted}


HISTORY_SUMMARY_PROMPT = """You are the Plate Coach weekly trend summarizer.
Here are the Healthy Scores and macro profiles for the user's last {n} meals, in order:
{meals_json}

Write a short (2-4 sentence) textual summary of any trend you notice (e.g.
declining vegetable intake, rising sugar, improving scores). Base your summary
only on the numbers given -- do not invent figures.
"""


def history_analyzer_node(state: PlateCoachState) -> dict:
    """Node 8: runs vision+nutrition+scoring over up to 20 stored photos,
    computes deterministic trend stats, and asks the LLM only for the prose summary."""
    photos = state.get("history_photos", [])[:20]

    meals = []
    for path in photos:
        try:
            vision_result = classify_image(path, top_k=1)
        except (FileNotFoundError, ValueError):
            continue
        food_names = [f["name"] for f in vision_result["foods"]]
        lookup = lookup_nutrition(food_names)
        profile = aggregate_profile(lookup["matched"], portion_size=vision_result.get("portion_size", "medium"))
        score = compute_healthy_score(profile)
        meals.append({"path": path, "score": score["score"], "veg_score": round(profile["veg_score"], 2), "sugar_g": round(profile["sugar_g"], 1)})

    if not meals:
        return {"history_summary": "No valid meal photos were found to analyze."}

    model = get_chat_model()
    prompt = HISTORY_SUMMARY_PROMPT.format(n=len(meals), meals_json=json.dumps(meals, indent=2))
    response = model.invoke(prompt)
    usage = extract_usage(response)

    token_usage = dict(state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0})
    token_usage["prompt_tokens"] += usage["prompt_tokens"]
    token_usage["completion_tokens"] += usage["completion_tokens"]

    return {"history_summary": response.content, "token_usage": token_usage}


def evaluator_node(state: PlateCoachState) -> dict:
    """Node 9: consistency + prescriptive-language check on the final output before
    it's returned to the user. Bounded 1-retry loop back to the planner (via a
    conditional edge) if something's off, rather than blocking outright."""
    suggestions = state.get("adjusted_suggestions") or state.get("suggestions", [])
    score = state.get("healthy_score", {}).get("score")
    retry_count = state.get("retry_count", 0)

    problems = []
    if score is None:
        problems.append("Missing Healthy Score.")
    if len(suggestions) < 2:
        problems.append("Fewer than 2 suggestions present.")
    for s in suggestions:
        flagged = find_prescriptive_language(s["text"])
        if flagged:
            problems.append(f"Prescriptive language in suggestion: {s['text']!r}")

    evaluator_ok = len(problems) == 0 or retry_count >= 1

    final_output = {
        "refused": False,
        "healthy_score": state.get("healthy_score"),
        "qualitative_profile": state.get("qualitative_profile"),
        "suggestions": suggestions,
    }
    final_output = apply_guardrails(final_output)

    return {
        "evaluator_ok": evaluator_ok,
        "evaluator_notes": "; ".join(problems) if problems else None,
        "retry_count": retry_count + 1,
        "final_output": final_output,
    }


def hitl_gate_node(state: PlateCoachState) -> dict:
    """Node 10: pauses the graph via LangGraph's native interrupt() before any
    persisted file write (history/profile), surfacing the pending payload for a
    [Y/N] confirmation. Resumed via Command(resume=bool) from the caller."""
    pending_write = state.get("pending_write")
    if not pending_write:
        return {}

    approved = interrupt({"action": "confirm_write", "payload": pending_write})
    if not approved:
        return {"pending_write": None, "error": "Write cancelled by user."}
    return {"pending_write": None}
