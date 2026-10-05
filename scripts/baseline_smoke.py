"""Run the harness unit checks, then a two-file TUEV eval and a resume of that same directory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from TUEV_eval import build_questions, find_rec_edf_pairs
from llm_settings import load_env

OUT_DIR = PROJECT_ROOT / "runs" / "tuev_baseline_smoke"
FILE_LIST = OUT_DIR / "smoke_files.txt"
PY = sys.executable
AUTHORS_TOOL_TEXT = "Use this tool to classify the type of artifact"


def fail(detail: str) -> None:
    """Stop the smoke test with a FAIL line."""
    raise SystemExit(f"[FAIL] {detail}")


def run_unit_tests() -> None:
    """Run the parser, scorer, and harness checks before any live eval."""
    for script in ("scripts/test_tuev_metrics.py", "scripts/test_harness.py", "scripts/test_parse_calling.py"):
        print(f"=== {script}")
        subprocess.check_call([PY, script], cwd=PROJECT_ROOT)


def choose_files(data_dir: str) -> list[str]:
    """Pick the two recordings with the fewest candidate windows."""
    pairs = find_rec_edf_pairs(data_dir)
    questions = build_questions(pairs)
    counts: dict[str, int] = {}
    for question in questions:
        counts[question["stem"]] = counts.get(question["stem"], 0) + 1
    ranked = sorted((count, stem) for stem, count in counts.items() if count > 0)
    if len(ranked) < 2:
        fail(f"Need two TUEV files with windows under {data_dir}")
    chosen = [stem for _count, stem in ranked[:2]]
    print("Smoke files:", ", ".join(f"{stem} ({counts[stem]} windows)" for stem in chosen))
    return chosen


def eval_command(file_list: Path, resume: bool) -> list[str]:
    """Build the TUEV_eval command for the smoke directory."""
    command = [
        PY,
        "TUEV_eval.py",
        "--file-list",
        str(file_list),
        "--out-dir",
        str(OUT_DIR),
        "--sleep",
        "0",
        "--seed",
        "0",
        "--protocol",
        "tuev_authors_v2",
    ]
    if resume:
        command.append("--resume")
    return command


def message_rows(out_dir: Path) -> list[dict]:
    """Read every per-window transcript written under the smoke directory."""
    rows = []
    for path in sorted(out_dir.glob("*/messages/*.messages.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_run(out_dir: Path) -> int:
    """Check the manifest, transcripts, stop behavior, and loaded context. Returns the row count."""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        fail("manifest.json was not written")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("prompt_sha256", "faiss_index_sha256", "chunks_sha256", "git", "sampling"):
        if not manifest.get(key):
            fail(f"manifest missing {key}")
    if manifest["protocol"]["harness"]["stop"] != ["<RETURN>"]:
        fail(f"manifest stop is {manifest['protocol']['harness']['stop']}")
    if manifest["sampling"]["seed"] != 0 or manifest["sampling"]["num_ctx"] != 65536:
        fail(f"manifest sampling is {manifest['sampling']}")
    planner = manifest.get("planner") or {}
    if not planner.get("digest"):
        fail(f"planner digest missing: {planner}")
    if planner.get("api") != "ollama_native":
        fail(f"planner api is {planner.get('api')}; /v1 would ignore num_ctx and top_k")
    if not manifest.get("authors_tool_schemas_sha256"):
        fail("manifest missing authors_tool_schemas_sha256")
    rows = message_rows(out_dir)
    if not rows:
        fail("no per-window message logs")
    calls = 0
    stop_hits = 0
    for row in rows:
        messages = row.get("messages")
        if not messages:
            fail(f"window {row.get('candidate_index')} has no transcript")
        if AUTHORS_TOOL_TEXT not in messages[0].get("content", ""):
            fail("system prompt does not show the authors' tool descriptions")
        for message in messages:
            content = message.get("content") or ""
            if message.get("role") == "assistant" and "<RETURN>" in content:
                fail("assistant message contains <RETURN>; the stop truncation did not hold")
            if "<NOTE>" in content:
                fail("authors prompt transcript contains a <NOTE> block")
    for raw_path in out_dir.glob("*/agent_raw.jsonl"):
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error"):
                fail(f"{record.get('stem')} window {record.get('candidate_index')} failed: {record['error']}")
            meta = record.get("result_meta") or {}
            for call in meta.get("calls") or []:
                calls += 1
                if call.get("stop") != ["<RETURN>"]:
                    fail(f"call was sent without the stop sequence: {call.get('stop')}")
                if call.get("api") != "ollama_native":
                    fail(f"call used api {call.get('api')}")
                if call.get("context_truncated") or call.get("finish_reason") == "context_overflow":
                    fail(f"context guard fired: {call}")
                if call.get("stop_hit"):
                    stop_hits += 1
                prompt_tokens = call.get("prompt_tokens")
                num_ctx = call.get("num_ctx") or manifest["sampling"]["num_ctx"]
                if prompt_tokens is None:
                    fail("a planner call did not report prompt_tokens")
                if prompt_tokens >= num_ctx:
                    fail(f"prompt_tokens {prompt_tokens} reached num_ctx {num_ctx}")
    if calls == 0:
        fail("no planner calls were logged")
    health_path = out_dir / "run_health.json"
    if not health_path.is_file():
        fail("run_health.json was not written")
    health = json.loads(health_path.read_text(encoding="utf-8"))
    if health.get("ollama_loaded_context") != manifest["sampling"]["num_ctx"]:
        fail(f"Ollama loaded context {health.get('ollama_loaded_context')}, expected {manifest['sampling']['num_ctx']}")
    print(
        f"[PASS] live: messages={len(rows)} calls={calls} stop_hits={stop_hits} "
        f"loaded_ctx={health['ollama_loaded_context']} max_prompt_tokens={health['max_prompt_tokens']}"
    )
    return len(rows)


def main() -> None:
    """Run the unit checks, a two-file eval, and a resume that must not drop transcripts."""
    load_env()
    run_unit_tests()
    data_dir = os.environ.get("TUEV_DATA_DIR")
    if not data_dir:
        fail("TUEV_DATA_DIR is not set")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = choose_files(data_dir)
    FILE_LIST.write_text("\n".join(stems) + "\n", encoding="utf-8")
    print("=== live eval")
    subprocess.check_call(eval_command(FILE_LIST, resume=False), cwd=PROJECT_ROOT)
    before = assert_run(OUT_DIR)
    print("=== resume")
    subprocess.check_call(eval_command(FILE_LIST, resume=True), cwd=PROJECT_ROOT)
    after_rows = message_rows(OUT_DIR)
    if len(after_rows) != before:
        fail(f"resume changed message rows from {before} to {len(after_rows)}")
    if any(not row.get("messages") for row in after_rows):
        fail("resume dropped a transcript")
    print(f"[PASS] resume kept {before} transcripts")
    print("Baseline smoke passed.")


if __name__ == "__main__":
    main()
