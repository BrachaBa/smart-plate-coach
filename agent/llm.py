"""OpenRouter LLM client wiring, shared by every node that needs an LLM call.

OpenRouter exposes an OpenAI-compatible API, so we use langchain_openai.ChatOpenAI
pointed at OpenRouter's base URL rather than a bespoke client. Model names are
configurable via .env / environment variables with sensible free+paid placeholder
defaults -- see .env.example. Swap the real model names in later; nothing here
should need to change when you do.
"""

import os
import ssl

import httpx
import truststore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Placeholder defaults: one free-tier model, one paid model (used e.g. by the cost eval,
# which the spec requires to include at least one paid model in its comparison).
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
PAID_MODEL = os.getenv("PAID_MODEL", "anthropic/claude-3.5-sonnet")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)


def _build_http_client() -> httpx.Client:
    """Use the OS-native trust store (truststore) instead of certifi's bundled CA
    list. Some networks run a TLS-intercepting proxy (e.g. a school/ISP content
    filter) whose root CA is trusted by the OS but isn't in certifi's bundle, or
    lacks the Key Usage extension Python 3.13+'s strict X.509 verification wants
    -- the OS trust store handles both cases the same way the browser/curl already
    do on that machine, with no bundled certs or verification flags to hardcode."""
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.Client(verify=ctx)


def get_chat_model(model: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at OpenRouter for the given model name."""
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        http_client=_build_http_client(),
        temperature=temperature,
    )


def extract_usage(response) -> dict:
    """Pull {"prompt_tokens": int, "completion_tokens": int} out of a ChatOpenAI response's
    usage_metadata, defaulting to zeros if the provider didn't report it."""
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
