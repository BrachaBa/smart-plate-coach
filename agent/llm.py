"""OpenRouter LLM client wiring, shared by every node that needs an LLM call.

OpenRouter exposes an OpenAI-compatible API, so we use langchain_openai.ChatOpenAI
pointed at OpenRouter's base URL rather than a bespoke client. Model names are
configurable via .env / environment variables -- see .env.example.

Every LLM call in the agent goes through call_llm(), which records one entry per
call in the state's usage_log ({node, model, prompt_tokens, completion_tokens}).
That log is what the CLI's cost summary and evals/cost_eval.py price per step.
"""

import os
import ssl
import sys
from pathlib import Path

import httpx
import openai
import truststore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def _load_env():
    """Load .env from the working directory, and -- for agent.exe -- from the
    folder the exe sits in, so a user can drop their .env next to the exe."""
    load_dotenv()
    if getattr(sys, "frozen", False):
        load_dotenv(Path(sys.executable).resolve().parent / ".env")


_load_env()

# LangSmith tracing is on when LANGSMITH_TRACING=true *and* a key is present.
# Tracing "on" without a key only makes the client fail in the background, so it
# is switched off here instead.
if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"
LANGSMITH_ENABLED = os.getenv("LANGSMITH_TRACING", "").lower() == "true"

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# One free-tier model for everyday use, one paid model (the cost eval must include
# a paid one), and a fixed judge for LLM-as-judge (kept separate from the models
# under test, so no model grades its own answers).
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "nvidia/nemotron-3.5-lightning:free")
# Free models are often rate-limited upstream (429). On a 429 from a ":free"
# model, call_llm retries the next free model here, and logs the one that answered.
FREE_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "FREE_FALLBACK_MODELS",
        "nvidia/nemotron-3.5-lightning:free,google/gemma-4-26b-a4b-it:free,google/gemma-4-31b-it:free",
    ).split(",")
    if m.strip()
]
PAID_MODEL = os.getenv("PAID_MODEL", "anthropic/claude-haiku-5.5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-5.5")
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "60"))


def _build_http_client() -> httpx.Client:
    """Use the OS-native trust store (truststore) instead of certifi's bundled CA
    list. Some networks run a TLS-intercepting proxy (e.g. a school/ISP content
    filter) whose root CA is trusted by the OS but isn't in certifi's bundle, or
    lacks the Key Usage extension Python 3.13+'s strict X.509 verification wants
    -- the OS trust store handles both cases the same way the browser/curl already
    do on that machine, with no bundled certs or verification flags to hardcode."""
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.Client(verify=ctx, timeout=REQUEST_TIMEOUT_S)


def get_chat_model(model: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at OpenRouter for the given model name."""
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        api_key=OPENROUTER_API_KEY or "missing",
        base_url=OPENROUTER_BASE_URL,
        http_client=_build_http_client(),
        temperature=temperature,
        max_retries=2,
        default_headers={"X-Title": "Plate Coach"},
    )


def extract_usage(response) -> dict:
    """Pull {"prompt_tokens": int, "completion_tokens": int} out of a ChatOpenAI response's
    usage_metadata, defaulting to zeros if the provider didn't report it."""
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


def call_llm(state: dict, node: str, prompt: str, temperature: float = 0.2) -> tuple[str, dict]:
    """Invoke the state's model (or DEFAULT_MODEL) and log the call's token usage.

    Returns (response_text, state_update) where state_update carries the
    extended usage_log and the running token_usage total -- merge it into the
    node's return dict.
    """
    requested = state.get("model") or DEFAULT_MODEL
    candidates = [requested]
    if requested.endswith(":free"):
        candidates += [m for m in FREE_FALLBACKS if m != requested]

    for i, model_name in enumerate(candidates):
        try:
            response = get_chat_model(model_name, temperature).invoke(prompt)
            break
        except openai.RateLimitError:
            if i == len(candidates) - 1:
                raise
    usage = extract_usage(response)

    usage_log = list(state.get("usage_log") or [])
    usage_log.append({"node": node, "model": model_name, **usage})
    total = {
        "prompt_tokens": sum(e["prompt_tokens"] for e in usage_log),
        "completion_tokens": sum(e["completion_tokens"] for e in usage_log),
    }
    return str(response.content), {"usage_log": usage_log, "token_usage": total}
