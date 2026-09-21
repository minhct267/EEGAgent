"""Load Ollama Cloud planner and local embedding settings from .env."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

DEFAULT_CLOUD_BASE_URL = "https://ollama.com/v1"
DEFAULT_PLANNER_MODEL = "qwen3.5:397b"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_EMBED_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_EMBED_API_KEY = "ollama"
DEFAULT_EMBED_MODEL = "bge-m3:latest"


@dataclass(frozen=True)
class PlannerSettings:
    api_key: str
    base_url: str
    model: str
    timeout: float
    reasoning_effort: str


@dataclass(frozen=True)
class EmbedSettings:
    api_key: str
    base_url: str
    model: str


def load_env() -> None:
    """Load the project .env once. Later calls are cheap."""
    load_dotenv(PROJECT_ROOT / ".env")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a number, got {raw!r}.") from exc


@lru_cache(maxsize=1)
def get_planner_settings() -> PlannerSettings:
    load_env()
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is missing. Set it in the project .env file before calling the planner."
        )
    return PlannerSettings(
        api_key=api_key,
        base_url=os.environ.get("OLLAMA_CLOUD_BASE_URL", DEFAULT_CLOUD_BASE_URL).strip()
        or DEFAULT_CLOUD_BASE_URL,
        model=os.environ.get("OLLAMA_PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip()
        or DEFAULT_PLANNER_MODEL,
        timeout=_env_float("OLLAMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        reasoning_effort=os.environ.get("OLLAMA_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
        or DEFAULT_REASONING_EFFORT,
    )


@lru_cache(maxsize=1)
def get_embed_settings() -> EmbedSettings:
    load_env()
    api_key = os.environ.get("OLLAMA_EMBED_API_KEY", DEFAULT_EMBED_API_KEY).strip() or DEFAULT_EMBED_API_KEY
    return EmbedSettings(
        api_key=api_key,
        base_url=os.environ.get("OLLAMA_EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL).strip()
        or DEFAULT_EMBED_BASE_URL,
        model=os.environ.get("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL,
    )


def planner_client() -> OpenAI:
    settings = get_planner_settings()
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.timeout)


def embed_client() -> OpenAI:
    settings = get_embed_settings()
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=60.0)


def strip_think_tags(text: str | None) -> str:
    """Remove leaked reasoning blocks so tool-call regex stays reliable."""
    if not text:
        return ""
    cleaned = THINK_TAG_RE.sub("", text)
    return cleaned.strip()


def has_planner_api_key() -> bool:
    load_env()
    return bool(os.environ.get("OLLAMA_API_KEY", "").strip())
