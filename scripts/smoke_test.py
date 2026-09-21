"""Smoke-test the planner (local Ollama by default) and local BGE-M3 embeddings.

Does not run TUEV / MDD / Sleep evaluation suites.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm_settings import (  # noqa: E402
    get_embed_settings,
    get_planner_settings,
    is_local_base_url,
    load_env,
    model_name_available,
    planner_client,
    planner_extra_body,
    planner_is_ready,
    strip_think_tags,
)
from RAG.embedder import BGEEmbedder  # noqa: E402
from RAG.searcher import FaissSearcher  # noqa: E402
from main import EEGAgent  # noqa: E402
from planner_adapt import DISCHARGE_TOOL_NAMES  # noqa: E402
from tools.dataLoad import dataLoad  # noqa: E402
from tools.registerData import registerData  # noqa: E402
from tools.singleChannel import seizureNormalModel_OneSecond  # noqa: E402
from tools.windowInfo import compute_amplitude  # noqa: E402
from utils.parseCalling import extract_tool_calls  # noqa: E402

SAMPLE_EDF = PROJECT_ROOT / "data" / "gped_049_a_6.edf"
INDEX_PATH = PROJECT_ROOT / "RAG" / "faiss.index"
CHUNKS_PATH = PROJECT_ROOT / "RAG" / "chunks.pkl"
REGISTRY_PATH = PROJECT_ROOT / "RAG" / "registered_files.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
AGENT_QUESTION = "Can epileptic discharges be observed within the first minute? If so, where?"
ALL_PHASES = ("env", "planner", "embed", "rag", "tools", "agent")
DEFAULT_PHASES = ("env", "planner", "embed", "rag", "tools")
PHASE_ALIASES = {"cloud": "planner"}


def _pass(name: str, detail: str) -> None:
    print(f"[PASS] {name}: {detail}")


def _fail(name: str, detail: str) -> None:
    print(f"[FAIL] {name}: {detail}")
    raise SystemExit(1)


def _list_local_models(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item.get("id", "") for item in payload.get("data", [])]


def phase_env() -> None:
    load_env()
    embed = get_embed_settings()
    planner = get_planner_settings()
    if not SAMPLE_EDF.exists():
        _fail("env", f"Sample EDF not found: {SAMPLE_EDF}")

    try:
        model_ids = _list_local_models(embed.base_url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        _fail(
            "env",
            f"Local Ollama is not reachable at {embed.base_url}. Start Ollama and pull {embed.model}. ({exc})",
        )

    if not model_name_available(model_ids, embed.model):
        _fail(
            "env",
            f"Embedding model {embed.model} is not available locally. Seen models: {model_ids}",
        )

    if is_local_base_url(planner.base_url) and not model_name_available(model_ids, planner.model):
        _fail(
            "env",
            f"Planner model {planner.model} is not available locally. Seen models: {model_ids}",
        )

    _pass(
        "env",
        f"Local Ollama is up, embed={embed.model}, planner={planner.model} at {planner.base_url}.",
    )


def phase_planner() -> None:
    ready, detail = planner_is_ready()
    if not ready:
        _fail("planner", detail)
    settings = get_planner_settings()
    client = planner_client()
    completion = client.chat.completions.create(
        model=settings.model,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
        timeout=settings.timeout,
        extra_body=planner_extra_body(settings),
    )
    text = strip_think_tags(completion.choices[0].message.content)
    if not text:
        _fail("planner", "Planner returned an empty message.")
    _pass("planner", f"{detail} reply={text!r}")


def phase_embed() -> None:
    embedder = BGEEmbedder()
    vectors = embedder.encode(
        [
            "epileptiform discharge",
            "epileptiform discharge",
            "The weather is sunny today.",
        ]
    )
    if len(vectors) != 3:
        _fail("embed", f"Expected 3 vectors, got {len(vectors)}")
    dim = len(vectors[0])
    if dim != 1024:
        _fail("embed", f"Expected 1024-d embeddings from bge-m3, got {dim}")
    self_sim = float(np.dot(vectors[0], vectors[1]))
    other_sim = float(np.dot(vectors[0], vectors[2]))
    if self_sim < 0.99:
        _fail("embed", f"Identical texts should have cosine ~1, got {self_sim:.4f}")
    if other_sim >= self_sim:
        _fail("embed", f"Unrelated text scored {other_sim:.4f} >= self-similarity {self_sim:.4f}")
    _pass("embed", f"dim={dim} self_sim={self_sim:.4f} unrelated_sim={other_sim:.4f}")


def phase_rag() -> None:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        _fail("rag", "FAISS index is missing. Run: python -m RAG.indexer --rebuild")
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
        if any(str(path).startswith("/data1/") for path in registry):
            _fail(
                "rag",
                "FAISS registry still points at the original Linux paths. Run: python -m RAG.indexer --rebuild",
            )

    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        threshold = json.load(handle).get("similarity_threshold", 0.6)

    embedder = BGEEmbedder()
    searcher = FaissSearcher(str(INDEX_PATH), str(CHUNKS_PATH))
    query_vector = embedder.encode(["epileptiform discharge GPED"])[0]
    results = searcher.search(query_vector, top_k=3)
    if not results:
        _fail("rag", "Search returned no chunks.")
    top_text, top_score = results[0]
    if top_score < threshold:
        _fail(
            "rag",
            f"Top score {top_score:.3f} < {threshold}. Rebuild the index with: python -m RAG.indexer --rebuild",
        )
    preview = " ".join(top_text.split())[:160]
    _pass("rag", f"top_score={top_score:.3f} preview={preview!r}")


def phase_tools() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    data = dataLoad(str(SAMPLE_EDF), config)
    registerData(data)
    amplitude = compute_amplitude(["FP1-F7", "F7-T3"], 0, 10, config)
    seizure = seizureNormalModel_OneSecond(["FP1-F7"], 88, 92, config)
    if not amplitude:
        _fail("tools", "compute_amplitude returned an empty result.")
    if not seizure:
        _fail("tools", "seizureNormalModel_OneSecond returned an empty result.")
    _pass("tools", f"amplitude_channels={list(amplitude)} seizure_windows={len(seizure)}")


def phase_agent() -> None:
    ready, detail = planner_is_ready()
    if not ready:
        _fail("agent", detail)
    agent = EEGAgent(
        config_path=str(CONFIG_PATH),
        file_name="gped_049_a_6.edf",
        tool_names=DISCHARGE_TOOL_NAMES,
    )
    result = agent.run(AGENT_QUESTION)
    response = result["response"] or ""
    called = any(
        extract_tool_calls(message.get("content") or "")
        for message in agent.messages
        if message.get("role") == "assistant"
    )
    if not called:
        _fail(
            "agent",
            f"Planner did not emit a parseable <FUNCTION> call. response={response[:300]!r}",
        )
    if not response.strip():
        _fail("agent", "Agent finished with an empty response.")
    print("Assistant:", response)
    _pass(
        "agent",
        f"rounds={result['rounds']} model_time={result['model_time']:.2f}s "
        f"local_tool_time={result['local_tool_time']:.2f}s total_time={result['total_time']:.2f}s",
    )


PHASE_FUNCS = {
    "env": phase_env,
    "planner": phase_planner,
    "embed": phase_embed,
    "rag": phase_rag,
    "tools": phase_tools,
    "agent": phase_agent,
}


def parse_phases(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(ALL_PHASES)
    phases = []
    for item in raw.split(","):
        name = PHASE_ALIASES.get(item.strip().lower(), item.strip().lower())
        if not name:
            continue
        phases.append(name)
    unknown = [item for item in phases if item not in PHASE_FUNCS]
    if unknown:
        raise SystemExit(f"Unknown phases: {unknown}. Choose from {list(ALL_PHASES)} or all.")
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test EEGAgent after the Ollama migration.")
    parser.add_argument(
        "--phase",
        default=",".join(DEFAULT_PHASES),
        help="Comma-separated phases: env,planner,embed,rag,tools,agent (or all). "
        "'cloud' is accepted as an alias for planner. Default skips the full agent call.",
    )
    args = parser.parse_args()
    phases = parse_phases(args.phase)
    print(f"Running phases: {', '.join(phases)}")
    for name in phases:
        PHASE_FUNCS[name]()
    print("All requested smoke-test phases passed.")


if __name__ == "__main__":
    main()
