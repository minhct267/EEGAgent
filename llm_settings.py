"""Load planner and embedding settings from .env.

The planner may be local Ollama or Ollama Cloud. Embeddings stay on local Ollama.
"""

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

DEFAULT_PLANNER_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_CLOUD_BASE_URL = "https://ollama.com/v1"
DEFAULT_PLANNER_MODEL = "qwen3.8:27b"
DEFAULT_PLANNER_API_KEY = "ollama"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_NUM_CTX = 16384
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
    num_ctx: int


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def is_local_base_url(base_url: str) -> bool:
    lowered = (base_url or "").lower()
    return "127.0.0.1" in lowered or "localhost" in lowered


def model_name_available(model_ids: list[str], model: str) -> bool:
    """Match Ollama tags with or without :latest."""
    names = {item for item in model_ids if item}
    if model in names:
        return True
    bare = model.replace(":latest", "")
    return bare in names or f"{bare}:latest" in names


@lru_cache(maxsize=1)
def get_planner_settings() -> PlannerSettings:
    load_env()
    base_url = (
        os.environ.get("OLLAMA_PLANNER_BASE_URL", "").strip()
        or os.environ.get("OLLAMA_CLOUD_BASE_URL", "").strip()
        or DEFAULT_PLANNER_BASE_URL
    )
    api_key = (
        os.environ.get("OLLAMA_PLANNER_API_KEY", "").strip()
        or os.environ.get("OLLAMA_API_KEY", "").strip()
        or DEFAULT_PLANNER_API_KEY
    )
    return PlannerSettings(
        api_key=api_key,
        base_url=base_url,
        model=os.environ.get("OLLAMA_PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip()
        or DEFAULT_PLANNER_MODEL,
        timeout=_env_float("OLLAMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        reasoning_effort=os.environ.get("OLLAMA_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
        or DEFAULT_REASONING_EFFORT,
        num_ctx=_env_int("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX),
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


def planner_extra_body(settings: PlannerSettings | None = None) -> dict:
    """Chat extras for Ollama. Older servers ignore unknown fields."""
    used = settings or get_planner_settings()
    body: dict = {"reasoning_effort": used.reasoning_effort}
    if used.reasoning_effort.strip().lower() in {"none", "off", "disabled"}:
        body["think"] = False
    if used.num_ctx > 0:
        body["options"] = {"num_ctx": used.num_ctx}
    return body


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
    """True when a planner key is set. Local Ollama accepts the dummy 'ollama' key."""
    load_env()
    return bool(
        os.environ.get("OLLAMA_PLANNER_API_KEY", "").strip()
        or os.environ.get("OLLAMA_API_KEY", "").strip()
        or DEFAULT_PLANNER_API_KEY
    )


def planner_is_ready() -> tuple[bool, str]:
    """Check that the planner endpoint is up and the configured model is listed."""
    settings = get_planner_settings()
    try:
        models = planner_client().models.list()
        model_ids = [item.id for item in models.data]
    except Exception as exc:
        return False, f"Planner is not reachable at {settings.base_url}: {exc}"
    if not model_name_available(model_ids, settings.model):
        return False, (
            f"Planner model {settings.model} was not in GET /v1/models at {settings.base_url}. "
            f"Seen: {model_ids}"
        )
    return True, f"model={settings.model} base={settings.base_url}"
