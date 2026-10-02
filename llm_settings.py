"""Load planner and embedding settings from .env (local Ollama by default)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import httpx
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
DEFAULT_NUM_CTX = 65536
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_SEED = 0
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
    temperature: float
    top_p: float
    top_k: int
    seed: int
    api: str = "openai"


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


def default_planner_api(base_url: str) -> str:
    """Ollama hosts use the native /api/chat. Its /v1 endpoint ignores options such as num_ctx."""
    lowered = (base_url or "").lower()
    return "ollama_native" if is_local_base_url(lowered) or "ollama.com" in lowered else "openai"


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
    api = os.environ.get("OLLAMA_PLANNER_API", "").strip() or default_planner_api(base_url)
    if api not in {"ollama_native", "openai"}:
        raise RuntimeError(f"OLLAMA_PLANNER_API must be ollama_native or openai, got {api!r}.")
    return PlannerSettings(
        api_key=api_key,
        base_url=base_url,
        model=os.environ.get("OLLAMA_PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip()
        or DEFAULT_PLANNER_MODEL,
        timeout=_env_float("OLLAMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        reasoning_effort=os.environ.get("OLLAMA_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
        or DEFAULT_REASONING_EFFORT,
        num_ctx=_env_int("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX),
        temperature=_env_float("OLLAMA_TEMPERATURE", DEFAULT_TEMPERATURE),
        top_p=_env_float("OLLAMA_TOP_P", DEFAULT_TOP_P),
        top_k=_env_int("OLLAMA_TOP_K", DEFAULT_TOP_K),
        seed=_env_int("OLLAMA_SEED", DEFAULT_SEED),
        api=api,
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
    """Chat extras for Ollama. Only the native API applies `options`; /v1 ignores them."""
    used = settings or get_planner_settings()
    body: dict = {"reasoning_effort": used.reasoning_effort}
    if used.reasoning_effort.strip().lower() in {"none", "off", "disabled"}:
        body["think"] = False
    options: dict = {
        "temperature": used.temperature,
        "top_p": used.top_p,
        "top_k": used.top_k,
        "seed": used.seed,
    }
    if used.num_ctx > 0:
        options["num_ctx"] = used.num_ctx
    body["options"] = options
    return body


class ContextOverflowError(RuntimeError):
    """Ollama refused a prompt longer than num_ctx (sent with truncate=false)."""

    def __init__(self, message: str, prompt_tokens: int | None, num_ctx: int | None):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.num_ctx = num_ctx


def ollama_root(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


class OllamaNativeClient:
    """`client.chat.completions.create()` over Ollama's /api/chat.

    /v1 drops `options`, so num_ctx and top_k never reach the runner there, and an
    over-long prompt silently loses its oldest messages. This client sends options
    natively and sets truncate=false so an overflow raises ContextOverflowError.
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.root = ollama_root(base_url)
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key and api_key != DEFAULT_PLANNER_API_KEY:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, timeout=None, temperature=None, top_p=None, seed=None,
               stop=None, max_tokens=None, extra_body=None, **_unused):
        extra = dict(extra_body or {})
        options = dict(extra.pop("options", None) or {})
        for key, value in (("temperature", temperature), ("top_p", top_p), ("seed", seed)):
            if value is not None:
                options[key] = value
        if stop:
            options["stop"] = list(stop)
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        effort = str(extra.get("reasoning_effort") or "").strip().lower()
        think = extra.get("think")
        if think is None:
            if effort in {"", "none", "off", "disabled"}:
                think = False
            else:
                think = effort if effort in {"low", "medium", "high", "xhigh"} else True
        payload = {
            "model": model,
            "messages": [{"role": item["role"], "content": item.get("content") or ""} for item in messages],
            "stream": False,
            "think": think,
            "truncate": False,
            "options": options,
        }
        response = httpx.post(
            f"{self.root}/api/chat",
            json=payload,
            headers=self.headers,
            timeout=timeout or self.timeout,
        )
        if response.status_code != 200:
            text = response.text
            if "exceed_context_size" in text or "exceeds the available context size" in text:
                prompt = re.search(r'n_prompt_tokens\\?"\s*:\s*(\d+)', text)
                ctx = re.search(r'n_ctx\\?"\s*:\s*(\d+)', text)
                raise ContextOverflowError(
                    text[:300],
                    int(prompt.group(1)) if prompt else None,
                    int(ctx.group(1)) if ctx else None,
                )
            raise RuntimeError(f"Ollama /api/chat returned {response.status_code}: {text[:300]}")
        data = response.json()
        message = data.get("message") or {}
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=message.get("content") or "", reasoning=message.get("thinking")),
                finish_reason=data.get("done_reason"),
            )],
            usage=SimpleNamespace(
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
            ),
        )


def make_planner_client(settings: PlannerSettings):
    if settings.api == "ollama_native":
        return OllamaNativeClient(settings.base_url, settings.api_key, settings.timeout)
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.timeout)


def planner_client():
    return make_planner_client(get_planner_settings())


def ollama_loaded_context(base_url: str, model: str) -> int | None:
    """context_length of a model currently loaded by Ollama, or None."""
    try:
        listed = httpx.get(f"{ollama_root(base_url)}/api/ps", timeout=10.0).json().get("models") or []
    except (httpx.HTTPError, ValueError):
        return None
    bare = model.replace(":latest", "")
    for item in listed:
        names = {(item.get("name") or "").replace(":latest", ""), (item.get("model") or "").replace(":latest", "")}
        if bare in names:
            return item.get("context_length")
    return None


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
        models = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.timeout).models.list()
        model_ids = [item.id for item in models.data]
    except Exception as exc:
        return False, f"Planner is not reachable at {settings.base_url}: {exc}"
    if not model_name_available(model_ids, settings.model):
        return False, (
            f"Planner model {settings.model} was not in GET /v1/models at {settings.base_url}. "
            f"Seen: {model_ids}"
        )
    return True, f"model={settings.model} base={settings.base_url}"
