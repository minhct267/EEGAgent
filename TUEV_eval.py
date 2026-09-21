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

from llm_settings import get_planner_settings, load_env
from main import EEGAgent
from utils.tuev_metrics import score_predictions, write_file_outputs, write_global_outputs

EVENT_PATTERN = re.compile(
    r"\(\s*([^,()]+?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)"
)

DEFAULT_DATA_DIR = r"D:\Datasets\TUH-EEG\TUEV\v2.0.1\edf\eval"
DEFAULT_OUT_DIR = "runs/tuev_agent_ollama"
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

WINDOW_QUESTION = """Please find all epileptic seizures in this EEG between {start} seconds and {end} seconds.
Check all channels. For each detected seizure, return exactly one line in this format:
(channel_name, time_start, time_end)
Do not include any extra text, explanation, or commentary.
Each line should correspond to one seizure event. List all events for all channels"""


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


def load_resumed_logs(out_dir, stem):
    path = Path(out_dir) / stem / "agent_raw.jsonl"
    if not path.exists():
        return set(), []
    done = set()
    logs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("messages", None)
            done.add(record["candidate_index"])
            logs.append(record)
    return done, logs


def append_raw_log(out_dir, stem, log):
    path = Path(out_dir) / stem / "agent_raw.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {key: value for key, value in log.items() if key != "messages"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(raw, ensure_ascii=False, default=str) + "\n")


def parse_events(raw_response):
    matches = EVENT_PATTERN.findall(raw_response or "")
    return [
        {"channel": channel.strip(), "start_time": float(start), "end_time": float(end)}
        for channel, start, end in matches
    ]


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


def run_eval(args):
    planner = get_planner_settings()
    pairs = find_rec_edf_pairs(args.data_dir, parse_file_list(args.file_list))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"No paired .edf/.rec files found under {args.data_dir}")

    questions = build_questions(pairs)
    ground_truth, negative_labels = load_ground_truth(pairs)
    grouped = defaultdict(list)
    for question in questions:
        grouped[question["edf"]].append(question)

    print(
        f"Evaluating {len(pairs)} files / {len(questions)} candidate windows "
        f"with model={planner.model} out_dir={args.out_dir}"
    )

    for pair in tqdm(pairs, desc="TUEV files"):
        edf_path = pair["edf"]
        file_questions = grouped.get(edf_path, [])
        done_indices, file_logs = load_resumed_logs(args.out_dir, pair["stem"]) if args.resume else (set(), [])
        if not args.resume:
            raw_path = Path(args.out_dir) / pair["stem"] / "agent_raw.jsonl"
            if raw_path.exists():
                raw_path.unlink()

        pending = [item for item in file_questions if item["candidate_index"] not in done_indices]
        if args.resume and not pending:
            print(f"Skipping completed file {pair['stem']} ({len(file_logs)} windows).")
        for question in pending:
            if args.sleep > 0:
                time.sleep(args.sleep)
            user_question = WINDOW_QUESTION.format(start=round(question["x"]), end=round(question["y"]))
            agent = None
            try:
                agent = EEGAgent(
                    config_path=args.config,
                    file_name=question["edf"],
                    api_key=planner.api_key,
                    base_url=planner.base_url,
                    model=planner.model,
                )
                result = agent.run(user_question)
                raw_response = result["response"]
                parsed_events = parse_events(raw_response)
                messages = agent.messages
                error = None
            except Exception as exc:
                result = None
                raw_response = ""
                parsed_events = []
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
                "raw_response": raw_response,
                "parsed_events": parsed_events,
                "result_meta": result,
                "messages_file": str(
                    Path(args.out_dir) / pair["stem"] / "messages" / f"{pair['stem']}.messages.json"
                ),
                "error": error,
                "messages": messages,
            }
            file_logs.append(log)
            append_raw_log(args.out_dir, pair["stem"], log)

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
        help="Output directory. Default does not overwrite runs/tuev_agent.",
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
        help="Skip candidate windows already recorded in agent_raw.jsonl.",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Seconds to wait before each new window.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
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
