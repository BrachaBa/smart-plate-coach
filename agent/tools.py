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
# Approved writes land in the working directory (next to agent.exe when run there).
HISTORY_PATH = Path("plate_coach_history.json")

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


_ARTICLES = ("a ", "an ", "the ", "some ", "extra ", "more ")


def normalize_food_name(name: str) -> str:
    """Canonical lookup key: lowercase, articles dropped, spaces/hyphens -> underscores.

    "A Side Salad" -> "side_salad", "french-fries" -> "french_fries".
    """
    key = (name or "").lower().strip()
    for article in _ARTICLES:
        if key.startswith(article):
            key = key[len(article):]
    return "_".join(key.replace("-", " ").split())


@lru_cache(maxsize=1)
def load_nutrition_db(path: Path = NUTRITION_DB_PATH) -> dict:
    """Load the nutrition-facts JSON once per process, keyed by normalized name and alias.

    Multiple keys (name + every alias) point at the same record dict.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    db = {}
    for record in records:
        db[normalize_food_name(record["name"])] = record
        for alias in record.get("aliases", []):
            db[normalize_food_name(alias)] = record
    return db


def find_food(name: str) -> dict | None:
    """Exact lookup on the normalized name, also trying the singular form.

    Deliberately no substring matching: "apple_pie" used to match "apple" (the
    fruit) and silently score a pie as fruit. A food that is not in the table is
    reported as unmatched instead of being matched to something it merely contains.
    """
    db = load_nutrition_db()
    key = normalize_food_name(name)
    if key in db:
        return db[key]
    if key.endswith("es") and key[:-2] in db:
        return db[key[:-2]]
    if key.endswith("s") and key[:-1] in db:
        return db[key[:-1]]
    return None


def lookup_nutrition(food_names: list) -> dict:
    """Resolve a list of food names to nutrition records (the RAG retrieval step).

    Returns {"matched": [{"query": str, **record}], "unmatched": [str, ...]}.
    Unmatched foods get a neutral default profile merged in (flagged via
    "is_default": True) rather than the LLM inventing plausible-looking numbers.
    """
    matched = []
    unmatched = []

    for query in food_names:
        record = find_food(query)
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
        # Drinks carry nutrients (a soda's sugar counts) but are not part of the
        # plate: averaging water's veg_score of 0 in made "add water" lower the score.
        if not record.get("is_drink"):
            veg_scores.append(record.get("veg_score", 0.0))

    for field in _MACRO_FIELDS:
        totals[field] *= multiplier

    totals["calories"] = totals.pop("calories_per_100g")
    totals["veg_score"] = sum(veg_scores) / len(veg_scores) if veg_scores else 0.0
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
    food = normalize_food_name(change.get("food") or "")
    replacement = change.get("replacement")

    after_records = list(meal_records)
    if action in ("remove", "swap") and food:
        after_records = [
            r for r in after_records
            if food not in (normalize_food_name(r.get("query", "")), normalize_food_name(r.get("name", "")))
        ]
    used_default = False
    if action in ("add", "swap") and replacement:
        added = lookup_nutrition([replacement])["matched"][0]
        used_default = added["is_default"]
        after_records.append(added)

    after_profile = aggregate_profile(after_records, portion_size)
    after_summary = _profile_summary(after_profile, compute_healthy_score(after_profile))

    delta = {key: round(after_summary[key] - before_summary[key], 1) for key in before_summary}

    # used_default: the added food is not in the database, so "after" rests on the
    # neutral default profile -- callers show no projected number in that case.
    return {"before": before_summary, "after": after_summary, "delta": delta, "used_default": used_default}


def find_foods_in_text(text: str) -> list:
    """Retrieval for free-text questions: every database food (name or alias)
    that appears in the text as whole words, longest match first, so
    "chocolate cake" wins over a bare "cake"-like alias."""
    db = load_nutrition_db()
    normalized = "_" + normalize_food_name(text.replace("?", " ").replace(",", " ").replace(".", " ")) + "_"
    found, taken = [], []
    for key in sorted(db, key=len, reverse=True):
        needle = f"_{key}_"
        start = normalized.find(needle)
        if start == -1:
            # also accept a plural form ("donuts" for "donut")
            start = normalized.find(f"_{key}s_")
        if start == -1:
            continue
        span = (start, start + len(needle))
        if any(a < span[1] and span[0] < b for a, b in taken):
            continue
        taken.append(span)
        name = db[key]["name"]
        if name not in found:
            found.append(name)
    return found


def perform_write(pending_write: dict) -> Path:
    """Execute an approved write (called only from the HITL node, after a yes).

    pending_write: {"target": str, "mode": "append_json" | "text", "content": ...}
    """
    target = Path(pending_write["target"]).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if pending_write.get("mode") == "append_json":
        entries = []
        if target.exists():
            try:
                entries = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                entries = []
        entries.append(pending_write["content"])
        target.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        target.write_text(str(pending_write["content"]), encoding="utf-8")
    return target.resolve()
