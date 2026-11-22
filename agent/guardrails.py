"""Cross-cutting ethical/safety guardrails (spec Part ד').

Binding or diagnostic phrasing gets rewritten into softer, informational wording
rather than blocked outright, and the fixed disclaimer is appended to every
non-refusal final output.
"""

import re

DISCLAIMER = (
    "This is general lifestyle information only, not professional medical or "
    "dietary advice. Consult a doctor or registered dietitian for personalized guidance."
)

_PRESCRIPTIVE_PATTERNS = [
    (re.compile(r"\byou must avoid\b", re.IGNORECASE), "you may want to consider limiting"),
    (re.compile(r"\byou are deficient in\b", re.IGNORECASE), "your meal appears relatively low in"),
    (re.compile(r"\byou have a deficiency in\b", re.IGNORECASE), "your meal appears relatively low in"),
    (re.compile(r"\byou need to\b", re.IGNORECASE), "you might consider"),
    (re.compile(r"\byou have to\b", re.IGNORECASE), "you could consider"),
    (re.compile(r"\byou must\b", re.IGNORECASE), "you may want to"),
    (re.compile(r"\byou should\b", re.IGNORECASE), "you could"),
]


def find_prescriptive_language(text: str) -> list:
    """Return which prescriptive patterns matched, if any.

    Used by the evaluator node to decide whether an automatic rewrite is needed
    before the output is considered final.
    """
    return [pattern.pattern for pattern, _ in _PRESCRIPTIVE_PATTERNS if pattern.search(text)]


def rewrite_prescriptive_language(text: str) -> str:
    """Replace binding/diagnostic phrasing with softer, non-prescriptive wording."""
    for pattern, replacement in _PRESCRIPTIVE_PATTERNS:
        text = pattern.sub(lambda m, r=replacement: r[0].upper() + r[1:] if m.group(0)[0].isupper() else r, text)
    return text


def apply_guardrails(final_output: dict) -> dict:
    """Rewrite prescriptive language and append the disclaimer.

    Refusal messages (out-of-scope replies) carry no dietary content, so they're
    left as-is -- the disclaimer only applies to responses that actually contain
    nutrition guidance.
    """
    if final_output.get("refused"):
        return final_output

    output = dict(final_output)
    if "suggestions" in output:
        output["suggestions"] = [
            {**s, "text": rewrite_prescriptive_language(s["text"])} for s in output["suggestions"]
        ]
    if "message" in output:
        output["message"] = rewrite_prescriptive_language(output["message"])
    if "qualitative_profile" in output:
        output["qualitative_profile"] = rewrite_prescriptive_language(output["qualitative_profile"])

    output["disclaimer"] = DISCLAIMER
    return output


# Deterministic security block, checked before any LLM sees the input: attempts
# to override the agent's instructions or extract its prompt are refused outright.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
        r"\bdisregard (all |any |the )?(previous|prior|above|your) (instructions|rules)",
        r"\b(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
        r"\byou are now\b",
        r"\bact as (?!a nutrition)",
        r"\bpretend (to be|you are)\b",
        r"\bjailbreak\b|\bDAN mode\b",
        r"\bdeveloper mode\b",
        r"<\s*/?\s*(system|user_input|question)\s*>",
    )
]


def detect_prompt_injection(text: str) -> bool:
    """True if the text tries to override instructions (prompt-injection guardrail)."""
    return any(p.search(text or "") for p in _INJECTION_PATTERNS)
