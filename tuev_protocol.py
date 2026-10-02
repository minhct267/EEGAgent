"""Frozen TUEV protocol: question, tools, harness, and sampling."""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from llm_settings import PlannerSettings, get_planner_settings, planner_extra_body
from planner_adapt import DISCHARGE_TOOL_NAMES, HarnessConfig, resolve_harness

PROJECT_ROOT = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_ROOT / "config" / "protocols"

AUTHORS_QUESTION = (
    "Please find all epileptic seizures in this EEG between {start} seconds and {end} seconds. \n"
    "Check all channels. For each detected seizure, return exactly one line in this format:\n"
    "(channel_name, time_start, time_end)\n"
    "Do not include any extra text, explanation, or commentary. \n"
    "Each line should correspond to one seizure event. List all events for all channels"
)

STRICT_QUESTION = """Please find all epileptic seizures in this EEG between {start} seconds and {end} seconds.
Check all channels. Inspect every second of this interval, starting at the beginning.
Report a channel only when a 1-second seizure tool shows seiz as the top class and seiz >= 0.50.
If consecutive seconds on the same channel meet that rule, return one merged line covering the full span.
For each detected seizure, return exactly one line in this format:
(channel_name, time_start, time_end)
Example:
(FP1-F7, 12.0, 14.0)
(FP2-F8, 12.5, 13.5)
If there are no seizures, write exactly: No events found
Do not include any extra text, explanation, or commentary."""


@dataclass(frozen=True)
class ResolvedProtocol:
    name: str
    question_template: str
    tool_names: tuple[str, ...] | None
    prompt_mode: str
    max_rounds: int
    harness: HarnessConfig
    temperature: float
    top_p: float
    top_k: int
    seed: int
    reasoning_effort: str
    num_ctx: int
    rag_enabled: bool
    rag_top_k: int
    source_path: str

    def sampling_overrides(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "reasoning_effort": self.reasoning_effort,
            "num_ctx": self.num_ctx,
        }

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "question_template": self.question_template,
            "tool_names": list(self.tool_names) if self.tool_names is not None else None,
            "prompt_mode": self.prompt_mode,
            "max_rounds": self.max_rounds,
            "harness": {
                "name": self.harness.name,
                "tool_result_role": self.harness.tool_result_role,
                "stop": list(self.harness.stop),
                "empty_retry": self.harness.empty_retry,
                "force_final": self.harness.force_final,
                "tool_calls": self.harness.tool_calls,
            },
            "sampling": self.sampling_overrides(),
            "rag": {"enabled": self.rag_enabled, "top_k": self.rag_top_k},
            "source_path": self.source_path,
        }


def protocol_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.resolve()
    named = PROTOCOL_DIR / f"{name_or_path}.json"
    if named.is_file():
        return named.resolve()
    raise FileNotFoundError(f"No protocol file at {candidate} or {named}.")


def load_protocol_file(name_or_path: str) -> dict:
    path = protocol_path(name_or_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_source_path"] = str(path)
    return payload


def _tool_names(raw, prompt_mode: str) -> tuple[str, ...] | None:
    if prompt_mode == "strict":
        return tuple(DISCHARGE_TOOL_NAMES)
    if raw is None:
        return None
    return tuple(raw)


def resolve_protocol(
    name_or_path: str = "tuev_authors_v2",
    harness: str | None = None,
    prompt_mode: str | None = None,
    rag: str | None = None,
    think: str | None = None,
    seed: int | None = None,
) -> ResolvedProtocol:
    """CLI flags override the protocol file. The file overrides environment sampling."""
    payload = load_protocol_file(name_or_path)
    sampling = payload.get("sampling") or {}
    rag_block = payload.get("rag") or {}
    mode = prompt_mode or payload.get("prompt_mode") or "authors"
    if mode not in {"authors", "strict"}:
        raise ValueError(f"Unknown prompt mode {mode!r}.")
    rag_enabled = bool(rag_block.get("enabled", True))
    if rag == "off":
        rag_enabled = False
    elif rag == "on":
        rag_enabled = True
    elif rag not in {None, ""}:
        raise ValueError(f"Unknown --rag value {rag!r}. Use on or off.")
    question = payload.get("question_template") or AUTHORS_QUESTION
    if mode == "strict":
        question = STRICT_QUESTION
    elif prompt_mode == "authors":
        question = payload.get("question_template") or AUTHORS_QUESTION
    harness_name = harness or payload.get("harness") or "authors_v2"
    return ResolvedProtocol(
        name=payload.get("name") or Path(payload["_source_path"]).stem,
        question_template=question,
        tool_names=_tool_names(payload.get("tool_names"), mode),
        prompt_mode=mode,
        max_rounds=int(payload.get("max_rounds") or 8),
        harness=resolve_harness(harness_name),
        temperature=float(sampling.get("temperature", 0.7)),
        top_p=float(sampling.get("top_p", 0.8)),
        top_k=int(sampling.get("top_k", 20)),
        seed=int(seed if seed is not None else sampling.get("seed", 0)),
        reasoning_effort=str(think if think is not None else sampling.get("reasoning_effort", "none")),
        num_ctx=int(sampling.get("num_ctx", 65536)),
        rag_enabled=rag_enabled,
        rag_top_k=int(rag_block.get("top_k", 3)),
        source_path=payload["_source_path"],
    )


def apply_protocol_settings(protocol: ResolvedProtocol, base: PlannerSettings | None = None) -> PlannerSettings:
    from dataclasses import replace

    settings = base or get_planner_settings()
    return replace(settings, **protocol.sampling_overrides())


def prompt_fingerprint(protocol: ResolvedProtocol) -> str:
    payload = protocol.as_dict()
    payload.pop("source_path", None)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path | None = None) -> dict:
    work = root or PROJECT_ROOT
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=work, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=work, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}
    return {"commit": commit, "dirty": bool(dirty)}


def _ollama_root(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def _ollama_json(base_url: str, path: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
    url = _ollama_root(base_url) + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _match_model(listed: dict, model: str) -> bool:
    names = {listed.get("name") or "", listed.get("model") or ""}
    bare = {item.replace(":latest", "") for item in names if item}
    return model in names or model.replace(":latest", "") in bare


def ollama_model_digest(base_url: str, model: str) -> dict:
    """Digest from /api/tags and quantization from /api/show. No credentials are read."""
    record = {"name": model, "base_url": base_url}
    try:
        tags = _ollama_json(base_url, "/api/tags")
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        record["error"] = str(exc)
        return record
    match = next((item for item in tags.get("models") or [] if _match_model(item, model)), None)
    if match:
        details = match.get("details") or {}
        record.update({
            "tag": match.get("name") or match.get("model"),
            "digest": match.get("digest"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "family": details.get("family"),
        })
    else:
        record["error"] = "model was not listed by /api/tags"
    try:
        shown = _ollama_json(base_url, "/api/show", {"name": model})
        shown_details = shown.get("details") or {}
        record["quantization"] = shown_details.get("quantization_level") or record.get("quantization")
        record["parameter_size"] = shown_details.get("parameter_size") or record.get("parameter_size")
        record["family"] = shown_details.get("family") or record.get("family")
        if shown.get("digest"):
            record["digest"] = shown.get("digest")
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        record["show_error"] = str(exc)
    return record


def build_manifest(protocol: ResolvedProtocol, settings: PlannerSettings, file_stems: list[str], config_path: str) -> dict:
    embed_settings = None
    try:
        from llm_settings import get_embed_settings

        embed_settings = get_embed_settings()
    except Exception as exc:
        embed_error = str(exc)
    else:
        embed_error = None
    index_path = PROJECT_ROOT / "RAG" / "faiss.index"
    chunks_path = PROJECT_ROOT / "RAG" / "chunks.pkl"
    manifest = {
        "protocol": protocol.as_dict(),
        "prompt_sha256": prompt_fingerprint(protocol),
        "sampling": protocol.sampling_overrides(),
        "planner": {
            "model": settings.model,
            "base_url": settings.base_url,
            "api": settings.api,
            "timeout_seconds": settings.timeout,
            **ollama_model_digest(settings.base_url, settings.model),
        },
        "embedder": None if embed_settings is None else {
            "model": embed_settings.model,
            "base_url": embed_settings.base_url,
            "error": embed_error,
            **ollama_model_digest(embed_settings.base_url, embed_settings.model),
        },
        "rag_enabled": protocol.rag_enabled,
        "faiss_index_sha256": sha256_file(index_path),
        "chunks_sha256": sha256_file(chunks_path),
        "authors_tool_schemas_sha256": sha256_file(PROTOCOL_DIR / "authors_tool_schemas.json")
        if protocol.prompt_mode == "authors" else None,
        "config_path": config_path,
        "config_sha256": sha256_file(Path(config_path) if Path(config_path).is_absolute() else PROJECT_ROOT / config_path),
        "git": git_state(),
        "file_stems": list(file_stems),
        "extra_body_keys": sorted(planner_extra_body(settings).keys()),
    }
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
