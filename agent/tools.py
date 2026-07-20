"""RAG (nutrition-facts lookup) and Code-Exec (deterministic scoring/delta) tools.

These are the only place numbers get computed in the agent. The LLM decides *which*
food was seen or *which* change to simulate; it never computes a score, calorie count,
or delta itself -- that would mean trusting free-text LLM arithmetic for a numeric
result the spec requires to be precise and reproducible. Every node that needs a
number calls into this module instead.
"""

import json
from functools import lru_cache
from pathlib import Path

NUTRITION_DB_PATH = Path(__file__).resolve().parent / "data" / "nutrition_facts.json"

# Per-meal reference thresholds used by the scoring formula below. These are general
# per-meal guidance figures, not a personalized medical/caloric prescription -- the
# agent never tailors these to an individual's body/health data.
PROTEIN_TARGET_G = 25.0
SUGAR_LIMIT_G = 25.0
SATFAT_LIMIT_G = 10.0
CARB_LIMIT_G = 75.0

PORTION_MULTIPLIER = {"small": 0.75, "medium": 1.0, "large": 1.4}

_MACRO_FIELDS = (
    "calories_per_100g",
    "protein_g",
    "carbs_g",
    "fat_g",
    "saturated_fat_g",
    "sugar_g",
)

_NEUTRAL_DEFAULT = {
    "calories_per_100g": 150.0,
    "protein_g": 5.0,
    "carbs_g": 15.0,
    "fat_g": 6.0,
    "saturated_fat_g": 2.0,
    "sugar_g": 3.0,
    "veg_score": 0.1,
    "serving_g": 200,
}


@lru_cache(maxsize=1)
def load_nutrition_db(path: Path = NUTRITION_DB_PATH) -> dict:
    """Load the nutrition-facts JSON once per process, keyed by lowercase name and alias.

    Multiple keys (name + every alias) point at the same record dict.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    db = {}
    for record in records:
        db[record["name"].lower()] = record
        for alias in record.get("aliases", []):
            db[alias.lower()] = record
    return db


def lookup_nutrition(food_names: list) -> dict:
    """Resolve a list of food names to nutrition records via exact match, falling
    back to a substring match, on the RAG database.

    Returns {"matched": [{"query": str, **record}], "unmatched": [str, ...]}.
    Unmatched foods get a neutral default profile merged in (flagged via
    "is_default": True) rather than the LLM inventing plausible-looking numbers.
    """
    db = load_nutrition_db()
    matched = []
    unmatched = []

    for query in food_names:
        key = query.lower().strip()
        record = db.get(key)

        if record is None:
            for db_key, db_record in db.items():
                if key in db_key or db_key in key:
                    record = db_record
                    break

        if record is not None:
            matched.append({"query": query, "is_default": False, **record})
        else:
            unmatched.append(query)
            matched.append({"query": query, "is_default": True, **_NEUTRAL_DEFAULT})

    return {"matched": matched, "unmatched": unmatched}


def aggregate_profile(matched_records: list, portion_size: str = "medium") -> dict:
    """Sum matched nutrition records into one meal-level profile, scaled by portion size.

    veg_score is averaged (it's a 0-1 qualitative proxy, not additive like grams).
    """
    multiplier = PORTION_MULTIPLIER.get(portion_size, 1.0)
    if not matched_records:
        return {field: 0.0 for field in _MACRO_FIELDS} | {"veg_score": 0.0}

    totals = {field: 0.0 for field in _MACRO_FIELDS}
    veg_scores = []
    for record in matched_records:
        serving_scale = record.get("serving_g", 200) / 100.0
        for field in _MACRO_FIELDS:
            totals[field] += record.get(field, 0.0) * serving_scale
        veg_scores.append(record.get("veg_score", 0.0))

    for field in _MACRO_FIELDS:
        totals[field] *= multiplier

    totals["calories"] = totals.pop("calories_per_100g")
    totals["veg_score"] = sum(veg_scores) / len(veg_scores)
    return totals


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def compute_healthy_score(profile: dict) -> dict:
    """Deterministic weighted Healthy Score (0-100, higher = healthier).

    profile: {"protein_g", "carbs_g", "fat_g", "saturated_fat_g", "sugar_g",
              "veg_score", "calories"} -- as produced by aggregate_profile().

    Weights: protein 0.25, vegetable content 0.20, carb-excess penalty 0.15,
    sugar penalty 0.20, saturated-fat penalty 0.20. Each sub-score is clipped to
    [0, 100] before weighting so no single extreme macro can push the total
    score out of range.
    """
    protein_score = _clip(profile.get("protein_g", 0.0) / PROTEIN_TARGET_G * 100)
    veg_score_pts = _clip(profile.get("veg_score", 0.0) * 100)

    carbs = profile.get("carbs_g", 0.0)
    excess_carbs_ratio = max(0.0, (carbs - CARB_LIMIT_G) / CARB_LIMIT_G)
    carb_score = _clip(100 - excess_carbs_ratio * 100)

    sugar_score = _clip(100 - profile.get("sugar_g", 0.0) / SUGAR_LIMIT_G * 100)
    satfat_score = _clip(100 - profile.get("saturated_fat_g", 0.0) / SATFAT_LIMIT_G * 100)

    score = (
        0.25 * protein_score
        + 0.20 * veg_score_pts
        + 0.15 * carb_score
        + 0.20 * sugar_score
        + 0.20 * satfat_score
    )

    return {
        "score": round(score),
        "breakdown": {
            "protein": round(protein_score),
            "veg": round(veg_score_pts),
            "carbs": round(carb_score),
            "sugar": round(sugar_score),
            "sat_fat": round(satfat_score),
        },
    }


def _profile_summary(profile: dict, score_result: dict) -> dict:
    return {
        "score": score_result["score"],
        "calories": round(profile.get("calories", 0.0), 1),
        "protein_g": round(profile.get("protein_g", 0.0), 1),
        "carbs_g": round(profile.get("carbs_g", 0.0), 1),
        "sugar_g": round(profile.get("sugar_g", 0.0), 1),
        "saturated_fat_g": round(profile.get("saturated_fat_g", 0.0), 1),
    }


def compute_delta(meal_records: list, change: dict, portion_size: str = "medium") -> dict:
    """Recompute the Healthy Score after a hypothetical "what if" change.

    meal_records: the list of matched nutrition records for the *current* meal
    (i.e. lookup_nutrition(...)["matched"]), not an already-aggregated profile --
    veg_score is a weighted average across components, not a scalar that can be
    incrementally patched, so this recomputes the aggregate from the component
    list before and after applying the change.

    change: {"action": "remove"|"add"|"swap", "food": str, "replacement": str|None}.
    "remove" drops `food` from meal_records. "add" appends `replacement` (or `food`
    if replacement is absent). "swap" drops `food` and appends `replacement`.

    Returns {"before": {...}, "after": {...}, "delta": {...}} where each of
    before/after is {"score": int, "calories": float, "protein_g": float, ...}
    and delta = after - before (signed) for score + every macro field.
    """
    before_profile = aggregate_profile(meal_records, portion_size)
    before_summary = _profile_summary(before_profile, compute_healthy_score(before_profile))

    action = change.get("action")
    food = (change.get("food") or "").lower()
    replacement = change.get("replacement")

    after_records = list(meal_records)
    if action in ("remove", "swap") and food:
        after_records = [r for r in after_records if r.get("query", r.get("name", "")).lower() != food]
    if action in ("add", "swap") and replacement:
        after_records.append(lookup_nutrition([replacement])["matched"][0])

    after_profile = aggregate_profile(after_records, portion_size)
    after_summary = _profile_summary(after_profile, compute_healthy_score(after_profile))

    delta = {key: round(after_summary[key] - before_summary[key], 1) for key in before_summary}

    return {"before": before_summary, "after": after_summary, "delta": delta}
