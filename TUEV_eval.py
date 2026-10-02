"""Evaluate EEGAgent on official TUEV eval pairs (.edf + .rec)."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from llm_settings import get_planner_settings, load_env, ollama_loaded_context
from main import EEGAgent
from tuev_protocol import (
    apply_protocol_settings,
    build_manifest,
    resolve_protocol,
    write_manifest,
)
from utils.tuev_metrics import score_predictions, write_file_outputs, write_global_outputs

EVENT_PATTERN = re.compile(
    r"\(\s*([^,()]+?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)"
)
LINE_EVENT_PATTERN = re.compile(
    r"^\s*[-*]?\s*([A-Za-z][A-Za-z0-9]*\s*-\s*[A-Za-z][A-Za-z0-9]*)\s*,"
    r"\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)

DEFAULT_DATA_DIR = "/home/nmduong/Data/datasets/TUH-EEG/TUEV/v2.0.1/edf/eval"
DEFAULT_OUT_DIR = "runs/tuev_authors_v2"
DEFAULT_CONFIG_PATH = "config/config.json"
DEFAULT_SLEEP_SECONDS = 5.0

REPORT_OVERLAP_THRESHOLD = 0.7
MERGE_GAP_THRESHOLD = 1.0
POSITIVE_CLASSES = [1, 2, 3]
NEGATIVE_CLASSES = [4, 5, 6]

CHANNEL_MAP = {
    0: "FP1-F7", 1: "F7-T3", 2: "T3-T5", 3: "T5-O1",
    4: "FP2-F8", 5: "F8-T4", 6: "T4-T6", 7: "T6-O2",
    8: "A1-T3", 9: "T3-C3", 10: "C3-CZ", 11: "CZ-C4",
    12: "C4-T4", 13: "T4-A2", 14: "FP1-F3", 15: "F3-C3",
    16: "C3-P3", 17: "P3-O1", 18: "FP2-F4", 19: "F4-C4",
    20: "C4-P4", 21: "P4-O2",
}

NO_EVENTS_RE = re.compile(r"no events found", re.IGNORECASE)
_CANONICAL_CHANNELS = {name.upper(): name for name in CHANNEL_MAP.values()}
_REVERSED_CHANNELS = {}
for _canonical in CHANNEL_MAP.values():
    _left, _right = _canonical.split("-")
    _REVERSED_CHANNELS[f"{_right}-{_left}"] = _canonical


def parse_file_list(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    return {item.strip() for item in raw.split(",") if item.strip()}


def find_rec_edf_pairs(data_dir, allowed=None):
    data_dir = Path(data_dir).resolve()
    rec_paths = sorted(data_dir.rglob("*.rec"))
    edf_by_stem = {path.stem: path.resolve() for path in data_dir.rglob("*.edf")}

    pairs = []
    for rec_path in rec_paths:
        rec_path = rec_path.resolve()
        edf_path = edf_by_stem.get(rec_path.stem)
        if not edf_path:
            continue
        if allowed and not (
            rec_path.stem in allowed or rec_path.name in allowed or edf_path.name in allowed
        ):
            continue
        pairs.append({
            "stem": rec_path.stem,
            "rec": str(rec_path),
            "edf": str(edf_path),
        })
    return pairs


def merge_rec_rows(input_file, gap_threshold=MERGE_GAP_THRESHOLD):
    df = pd.read_csv(input_file, header=None, names=["channel", "start", "end", "class"])
    df["channel"] = df["channel"].astype(int)
    df["class"] = df["class"].astype(int)
    merged = []
    for (channel, cls), sub in df.groupby(["channel", "class"]):
        current_start, current_end = None, None
        for _, row in sub.sort_values("start").iterrows():
            start, end = row["start"], row["end"]
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end + gap_threshold:
                current_end = max(current_end, end)
            else:
                merged.append((channel, current_start, current_end, cls))
                current_start, current_end = start, end
        if current_start is not None:
            merged.append((channel, current_start, current_end, cls))
    return pd.DataFrame(merged, columns=["channel", "start", "end", "class"])


def candidate_windows(merged_df, gap_threshold=MERGE_GAP_THRESHOLD):
    windows = []
    current_start, current_end = None, None
    for start, end in merged_df[["start", "end"]].sort_values("start").values.tolist():
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end + gap_threshold:
            current_end = max(current_end, end)
        else:
            windows.append((current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None:
        windows.append((current_start, current_end))
    return windows


def build_questions(pairs):
    questions = []
    for pair in pairs:
        merged_df = merge_rec_rows(pair["rec"])
        if merged_df.empty:
            continue
        for index, (start, end) in enumerate(candidate_windows(merged_df)):
            questions.append({
                "stem": pair["stem"],
                "edf": pair["edf"],
                "rec": pair["rec"],
                "candidate_index": index,
                "x": start,
                "y": end,
            })
    return questions


def load_ground_truth(pairs):
    gt_data = defaultdict(list)
    negative_data = defaultdict(list)
    for pair in pairs:
        merged_df = merge_rec_rows(pair["rec"])
        if merged_df.empty:
            continue
        positive = merged_df[merged_df["class"].isin(POSITIVE_CLASSES)].copy()
        positive["channel_name"] = positive["channel"].map(CHANNEL_MAP)
        positive.dropna(subset=["channel_name"], inplace=True)
        gt_data[pair["edf"]] = positive.to_dict("records")
        negative = merged_df[merged_df["class"].isin(NEGATIVE_CLASSES)].copy()
        negative["channel_name"] = negative["channel"].map(CHANNEL_MAP)
        negative.dropna(subset=["channel_name"], inplace=True)
        negative_data[pair["edf"]] = negative.to_dict("records")
    return gt_data, negative_data


def messages_jsonl_path(out_dir, stem):
    return Path(out_dir) / stem / "messages" / f"{stem}.messages.jsonl"


def load_message_index(out_dir, stem):
    """Restore transcripts. The jsonl written per window is the source of truth."""
    found = {}
    path = messages_jsonl_path(out_dir, stem)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                found[record["candidate_index"]] = record.get("messages")
        return found
    snapshot = Path(out_dir) / stem / "messages" / f"{stem}.messages.json"
    if not snapshot.exists():
        return found
    blob = json.loads(snapshot.read_text(encoding="utf-8"))
    for candidate in blob.get("candidates") or []:
        if candidate.get("messages"):
            found[candidate["candidate_index"]] = candidate["messages"]
    return found


def load_resumed_logs(out_dir, stem):
    path = Path(out_dir) / stem / "agent_raw.jsonl"
    if not path.exists():
        return set(), []
    messages = load_message_index(out_dir, stem)
    done = set()
    logs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            index = record["candidate_index"]
            transcript = messages.get(index)
            # A raw line with no saved transcript is run again. The file rewrite drops the stale line.
            if transcript is None and not record.get("error"):
                continue
            record["messages"] = transcript
            if index in done:
                logs = [item for item in logs if item["candidate_index"] != index]
            done.add(index)
            logs.append(record)
    return done, logs


def append_raw_log(out_dir, stem, log):
    path = Path(out_dir) / stem / "agent_raw.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {key: value for key, value in log.items() if key != "messages"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(raw, ensure_ascii=False, default=str) + "\n")


def append_messages_log(out_dir, stem, log):
    path = messages_jsonl_path(out_dir, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "candidate_index": log["candidate_index"],
        "window": log.get("window"),
        "question": log.get("question"),
        "model": log.get("model"),
        "messages": log.get("messages"),
        "parsed_events": log.get("parsed_events"),
        "invalid_events": log.get("invalid_events"),
        "format_status": log.get("format_status"),
        "error": log.get("error"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _normalize_channel(name):
    return re.sub(r"\s+", "", name).strip().upper()


def canonicalize_channel(name):
    """Map case, spaces, and reversed TCP pairs onto CHANNEL_MAP names."""
    cleaned = _normalize_channel(name)
    if cleaned in _CANONICAL_CHANNELS:
        return _CANONICAL_CHANNELS[cleaned], True
    if cleaned in _REVERSED_CHANNELS:
        return _REVERSED_CHANNELS[cleaned], True
    return cleaned, False


def parse_events_detailed(raw_response):
    text = raw_response or ""
    events = []
    invalid = []
    seen = set()
    for channel, start, end in EVENT_PATTERN.findall(text) + LINE_EVENT_PATTERN.findall(text):
        start_time = float(start)
        end_time = float(end)
        canonical, ok = canonicalize_channel(channel)
        key = (canonical, start_time, end_time, ok)
        if not canonical or key in seen:
            continue
        seen.add(key)
        item = {"channel": canonical, "start_time": start_time, "end_time": end_time}
        if ok:
            events.append(item)
        else:
            item["invalid_channel"] = True
            item["raw_channel"] = _normalize_channel(channel)
            invalid.append(item)
    return events, invalid


def parse_events(raw_response):
    events, _invalid = parse_events_detailed(raw_response)
    return events


def classify_answer(text, events, stopped_by_max_rounds):
    body = (text or "").strip()
    if events:
        return "tuples"
    if not body:
        return "max_rounds" if stopped_by_max_rounds else "empty"
    if NO_EVENTS_RE.search(body):
        return "no_events"
    if stopped_by_max_rounds:
        return "max_rounds"
    return "unparseable"


def write_file_metrics(out_dir, edf_path, rec_path, stem, model, raw_logs, ground_truth, negative_labels):
    raw_preds = [event for log in raw_logs for event in log.get("parsed_events", [])]
    file_metrics, _episodes, scored_episodes = score_predictions(
        ground_truth.get(edf_path, []),
        negative_labels.get(edf_path, []),
        raw_preds,
        threshold=REPORT_OVERLAP_THRESHOLD,
        gap_threshold=MERGE_GAP_THRESHOLD,
    )
    write_file_outputs(
        out_dir=out_dir,
        stem=stem,
        edf_path=edf_path,
        rec_path=rec_path,
        model=model,
        gt_events=ground_truth.get(edf_path, []),
        negative_events=negative_labels.get(edf_path, []),
        raw_predictions=raw_preds,
        metrics=file_metrics,
        scored_report_episodes=scored_episodes,
        raw_logs=raw_logs,
        report_overlap_threshold=REPORT_OVERLAP_THRESHOLD,
    )
    return file_metrics


def print_aggregate(aggregate):
    print("Analysis complete.")
    print(f"Files: {aggregate.get('files', 0)}")
    print(f"Total report episodes: {aggregate.get('total_reports', 0)}")
    print(f"Total ground truths: {aggregate.get('total_gt', 0)}")
    print(f"Hits (TP): {aggregate.get('hits', 0)}")
    print(f"Misses (FN): {aggregate.get('misses', 0)}")
    print(f"Hit rate: {aggregate.get('hit_rate', float('nan'))}")
    print(f"Strict unmatched reports: {aggregate.get('strict_unmatched_reports', 0)}")
    print(f"Strict unmatched report rate: {aggregate.get('strict_unmatched_report_rate', float('nan'))}")
    print(f"Explicit negative reports: {aggregate.get('explicit_negative_reports', 0)}")
    print(f"Explicit negative report rate: {aggregate.get('explicit_negative_report_rate', float('nan'))}")
    print(f"Unverified reports: {aggregate.get('unverified_reports', 0)}")
    print(f"Unverified report rate: {aggregate.get('unverified_report_rate', float('nan'))}")


def _select_answer(raw_response, result):
    events, invalid = parse_events_detailed(raw_response)
    answer_text = raw_response or ""
    fallback = (result or {}).get("raw_assistant") or ""
    if not events and fallback and fallback != answer_text:
        fallback_events, fallback_invalid = parse_events_detailed(fallback)
        if fallback_events or not answer_text.strip():
            events, invalid = fallback_events, fallback_invalid
            answer_text = fallback
    if not events and result and result.get("context_overflow"):
        return events, invalid, "context_overflow"
    stopped = bool(result and result.get("stopped_by_max_rounds") and not events)
    return events, invalid, classify_answer(answer_text, events, stopped)


def write_run_health(out_dir, planner):
    """Per-run counts of answer status and harness events, plus the context Ollama loaded."""
    status_counts = defaultdict(int)
    event_counts = defaultdict(int)
    windows = errors = calls = 0
    max_prompt_tokens = 0
    overflow_windows = []
    truncated_windows = []
    for raw_path in sorted(Path(out_dir).glob("*/agent_raw.jsonl")):
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            log = json.loads(line)
            windows += 1
            errors += bool(log.get("error"))
            status_counts[log.get("format_status") or "unknown"] += 1
            meta = log.get("result_meta") or {}
            events = set(meta.get("harness_events") or [])
            for event in events:
                event_counts[event] += 1
            window_id = f"{log.get('stem')}#{log.get('candidate_index')}"
            if "context_overflow" in events:
                overflow_windows.append(window_id)
            if "context_truncated" in events:
                truncated_windows.append(window_id)
            for call in meta.get("calls") or []:
                calls += 1
                max_prompt_tokens = max(max_prompt_tokens, call.get("prompt_tokens") or 0)
    health = {
        "windows": windows,
        "errors": errors,
        "planner_calls": calls,
        "format_status": dict(sorted(status_counts.items())),
        "windows_with_event": dict(sorted(event_counts.items())),
        "max_prompt_tokens": max_prompt_tokens,
        "num_ctx_requested": planner.num_ctx,
        "planner_api": planner.api,
        "ollama_loaded_context": ollama_loaded_context(planner.base_url, planner.model)
        if planner.api == "ollama_native" else None,
        "context_overflow_windows": overflow_windows,
        "context_truncated_windows": truncated_windows,
    }
    path = Path(out_dir) / "run_health.json"
    path.write_text(json.dumps(health, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Run health: windows={windows} errors={errors} max_prompt_tokens={max_prompt_tokens} "
        f"loaded_ctx={health['ollama_loaded_context']} overflow={len(overflow_windows)} "
        f"truncated={len(truncated_windows)} status={health['format_status']}"
    )
    return health


def run_eval(args):
    protocol = resolve_protocol(
        name_or_path=args.protocol,
        harness=args.harness,
        prompt_mode=args.prompt,
        rag=args.rag,
        think=args.think,
        seed=args.seed,
    )
    planner = apply_protocol_settings(protocol, get_planner_settings())
    pairs = find_rec_edf_pairs(args.data_dir, parse_file_list(args.file_list))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"No paired .edf/.rec files found under {args.data_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(protocol, planner, [pair["stem"] for pair in pairs], args.config)
    write_manifest(out_dir / "manifest.json", manifest)

    questions = build_questions(pairs)
    ground_truth, negative_labels = load_ground_truth(pairs)
    grouped = defaultdict(list)
    for question in questions:
        grouped[question["edf"]].append(question)

    print(
        f"Evaluating {len(pairs)} files / {len(questions)} candidate windows "
        f"protocol={protocol.name} harness={protocol.harness.name} prompt={protocol.prompt_mode} "
        f"rag={'on' if protocol.rag_enabled else 'off'} seed={protocol.seed} "
        f"think={protocol.reasoning_effort} model={planner.model} api={planner.api} "
        f"num_ctx={planner.num_ctx} out_dir={args.out_dir}"
    )

    for pair in tqdm(pairs, desc="TUEV files"):
        edf_path = pair["edf"]
        file_questions = grouped.get(edf_path, [])
        done_indices, file_logs = load_resumed_logs(args.out_dir, pair["stem"]) if args.resume else (set(), [])
        if not args.resume:
            raw_path = Path(args.out_dir) / pair["stem"] / "agent_raw.jsonl"
            if raw_path.exists():
                raw_path.unlink()
            messages_path = messages_jsonl_path(args.out_dir, pair["stem"])
            if messages_path.exists():
                messages_path.unlink()

        pending = [item for item in file_questions if item["candidate_index"] not in done_indices]
        if args.resume and not pending:
            print(f"Skipping completed file {pair['stem']} ({len(file_logs)} windows).")
        for question in pending:
            if args.sleep > 0:
                time.sleep(args.sleep)
            user_question = protocol.question_template.format(
                start=round(question["x"]), end=round(question["y"])
            )
            agent = None
            try:
                agent = EEGAgent(
                    config_path=args.config,
                    file_name=question["edf"],
                    api_key=planner.api_key,
                    base_url=planner.base_url,
                    model=planner.model,
                    tool_names=protocol.tool_names,
                    harness=protocol.harness,
                    prompt_mode=protocol.prompt_mode,
                    rag_enabled=protocol.rag_enabled,
                    rag_top_k=protocol.rag_top_k,
                    sampling=protocol.sampling_overrides(),
                )
                result = agent.run(user_question, max_rounds=protocol.max_rounds)
                raw_response = result["response"]
                parsed_events, invalid_events, format_status = _select_answer(raw_response, result)
                messages = agent.messages
                error = None
            except Exception as exc:
                result = None
                raw_response = ""
                parsed_events = []
                invalid_events = []
                format_status = "empty"
                messages = getattr(agent, "messages", None)
                error = repr(exc)
                print(f"{pair['stem']} window {question['candidate_index']} failed: {error}")

            log = {
                "pair_index": 0,
                "stem": pair["stem"],
                "edf": question["edf"],
                "rec": question["rec"],
                "candidate_index": question["candidate_index"],
                "window": {"start": question["x"], "end": question["y"]},
                "question": user_question,
                "model": planner.model,
                "protocol": protocol.name,
                "harness": protocol.harness.name,
                "prompt_mode": protocol.prompt_mode,
                "seed": protocol.seed,
                "raw_response": raw_response,
                "parsed_events": parsed_events,
                "invalid_events": invalid_events,
                "format_status": format_status,
                "result_meta": result,
                "messages_file": str(messages_jsonl_path(args.out_dir, pair["stem"])),
                "error": error,
                "messages": messages,
            }
            file_logs.append(log)
            append_raw_log(args.out_dir, pair["stem"], log)
            append_messages_log(args.out_dir, pair["stem"], log)

        write_file_metrics(
            args.out_dir,
            edf_path,
            pair["rec"],
            pair["stem"],
            planner.model,
            file_logs,
            ground_truth,
            negative_labels,
        )

    aggregate = write_global_outputs(args.out_dir)
    print_aggregate(aggregate)
    write_run_health(args.out_dir, planner)
    return aggregate


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Evaluate EEGAgent on official TUEV eval .edf/.rec pairs.")
    parser.add_argument("--mode", choices=["list", "eval"], default="eval")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("TUEV_DATA_DIR", DEFAULT_DATA_DIR),
        help="Directory containing paired TUEV .edf/.rec files.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("TUEV_OUT_DIR", DEFAULT_OUT_DIR),
        help="Output directory. Defaults to runs/tuev_authors_v2 so runs/tuev_agent_ollama is left intact.",
    )
    parser.add_argument(
        "--file-list",
        default=None,
        help="Comma-separated stems/filenames, or a path to a text file of those names.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N paired files.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip candidate windows already recorded in agent_raw.jsonl and reload their transcripts.",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Seconds to wait before each new window.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--protocol", default="tuev_authors_v2", help="Protocol name or path to a protocol JSON file.")
    parser.add_argument("--harness", choices=["authors_v1", "authors_v2"], default=None, help="Override the protocol harness.")
    parser.add_argument("--prompt", choices=["authors", "strict"], default=None, help="authors: original question and all tools. strict: 0.50 question and discharge tools.")
    parser.add_argument("--rag", choices=["on", "off"], default=None, help="Override protocol retrieval.")
    parser.add_argument("--think", default=None, help="Reasoning effort override, for example none or medium.")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed override.")
    args = parser.parse_args()

    pairs = find_rec_edf_pairs(args.data_dir, parse_file_list(args.file_list))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Found {len(pairs)} paired .rec/.edf files under {args.data_dir}")
    for pair in pairs:
        print(f"{pair['stem']}: rec={pair['rec']} edf={pair['edf']}")

    if args.mode == "list":
        return

    run_eval(args)


if __name__ == "__main__":
    main()
