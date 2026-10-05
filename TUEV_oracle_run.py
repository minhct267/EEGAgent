"""Tool-only TUEV ceiling on the agent's windows and scorer; a second is positive when seiz meets the threshold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from TUEV_eval import (
    build_questions,
    find_rec_edf_pairs,
    load_ground_truth,
    parse_file_list,
)
from llm_settings import load_env
from utils.tuev_metrics import aggregate_summaries, score_predictions, summarize_metrics, write_json

CHANNEL_NAMES = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-CZ", "CZ-C4",
    "C4-T4", "T4-A2", "FP1-F3", "F3-C3",
    "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
    "C4-P4", "P4-O2",
]
RULES = ("seizNormal", "seizArtiBckg")


def load_config(config_path):
    """Load the project config JSON used by the detection tools."""
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def asked_bounds(start, end):
    """Match TUEV_eval question text, which inserts round(start) and round(end)."""
    asked_start = int(round(start))
    asked_end = int(round(end))
    if asked_end <= asked_start:
        asked_end = asked_start + 1
    return asked_start, asked_end


def tool_for(rule: str):
    """Return the 1-second seizure tool selected by --rule."""
    if rule == "seizNormal":
        from tools.singleChannel import seizureNormalModel_OneSecond

        return seizureNormalModel_OneSecond
    if rule == "seizArtiBckg":
        from tools.singleChannel import seizureArtiBckgModel_OneSecond

        return seizureArtiBckgModel_OneSecond
    raise ValueError(f"Unknown rule {rule!r}. Choose from {RULES}.")


def predictions_from_result(result, threshold):
    """Keep channel-seconds whose seiz probability is at least the threshold."""
    predictions = []
    for item in result:
        duration = str(item["duration"]).replace("s", "")
        sec_start, sec_end = [int(value) for value in duration.split("-")]
        for channel in CHANNEL_NAMES:
            scores = item.get(channel) or {}
            probability = scores.get("seiz")
            if probability is None or probability < threshold:
                continue
            predictions.append({
                "channel": channel,
                "start_time": float(sec_start),
                "end_time": float(sec_end),
                "max_prob": probability,
            })
    return predictions


def run_oracle(pairs, config_path, threshold, rule, out_dir):
    """Run one tool over every asked window and score it like the agent."""
    from tools.dataLoad import dataLoad
    from tools.registerData import registerData

    config = load_config(config_path)
    detect = tool_for(rule)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = build_questions(pairs)
    ground_truth, negative_labels = load_ground_truth(pairs)
    grouped = {}
    for question in questions:
        grouped.setdefault(question["edf"], []).append(question)

    summaries = []
    with (out_dir / "oracle_raw.jsonl").open("w", encoding="utf-8") as raw_f, (
        out_dir / "oracle_predictions.jsonl"
    ).open("w", encoding="utf-8") as pred_f:
        for pair in pairs:
            registerData(dataLoad(pair["edf"], config))
            predictions = []
            for question in grouped.get(pair["edf"], []):
                start, end = asked_bounds(question["x"], question["y"])
                for chunk_start in range(start, end, 10):
                    chunk_end = min(chunk_start + 10, end)
                    result = detect(
                        name=list(CHANNEL_NAMES),
                        start=chunk_start,
                        end=chunk_end,
                        config=config,
                    )
                    raw_f.write(json.dumps({
                        "stem": pair["stem"],
                        "edf": pair["edf"],
                        "candidate_index": question["candidate_index"],
                        "window": {"start": question["x"], "end": question["y"]},
                        "asked": {"start": start, "end": end},
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "rule": rule,
                        "threshold": threshold,
                        "result": result,
                    }, ensure_ascii=False) + "\n")
                    predictions.extend(predictions_from_result(result, threshold))
            gt_events = ground_truth.get(pair["edf"], [])
            negative_events = negative_labels.get(pair["edf"], [])
            metrics, _episodes, scored = score_predictions(gt_events, negative_events, predictions)
            summary = summarize_metrics(metrics, f"oracle:{rule}@{threshold:g}")
            summary["stem"] = pair["stem"]
            summaries.append(summary)
            pred_f.write(json.dumps({
                "stem": pair["stem"],
                "edf": pair["edf"],
                "rec": pair["rec"],
                "rule": rule,
                "threshold": threshold,
                "gt_events": gt_events,
                "negative_events": negative_events,
                "predictions": predictions,
                "scored_report_episodes": scored,
                "metrics": metrics,
            }, ensure_ascii=False, default=str) + "\n")

    aggregate = aggregate_summaries(summaries) if summaries else {}
    aggregate["rule"] = rule
    aggregate["threshold"] = threshold
    aggregate["window_policy"] = "round_question_bounds"
    write_json(out_dir / "metrics.json", {"aggregate": aggregate, "per_file": summaries})
    write_json(out_dir / "summary.json", aggregate)
    return aggregate


def expand_rules(raw: str) -> list[str]:
    """Turn --rule both into the two seizure tools, or keep a single rule."""
    if raw == "both":
        return list(RULES)
    if raw not in RULES:
        raise SystemExit(f"Unknown --rule {raw!r}. Choose seizNormal, seizArtiBckg, or both.")
    return [raw]


def main():
    """List the paired files, or run the selected rule and threshold grid."""
    load_env()
    parser = argparse.ArgumentParser(description="Tool-only TUEV oracle aligned with the agent scorer.")
    parser.add_argument("--mode", choices=["list", "oracle"], default="oracle")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("TUEV_DATA_DIR"),
        help="Directory containing paired .edf/.rec files.",
    )
    parser.add_argument("--file-list", default=None, help="Comma-separated stems or a path to a stem list.")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument(
        "--rule",
        default="seizNormal",
        help="seizNormal, seizArtiBckg, or both.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=None,
        help="Repeat to run several thresholds. Default is 0.7. Use 0.5 and 0.7 for the baseline grid.",
    )
    parser.add_argument("--out-dir", default="runs/tuev_oracle")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.data_dir:
        raise SystemExit("Pass --data-dir or set TUEV_DATA_DIR.")

    pairs = find_rec_edf_pairs(args.data_dir, parse_file_list(args.file_list))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Found {len(pairs)} paired .rec/.edf files.")
    for pair in pairs:
        print(f"{pair['stem']}: rec={pair['rec']} edf={pair['edf']}")
    if args.mode == "list":
        return

    rules = expand_rules(args.rule)
    thresholds = args.threshold or [0.7]
    combos = [(rule, threshold) for rule in rules for threshold in thresholds]
    aggregates = []
    for rule, threshold in combos:
        dest = Path(args.out_dir)
        if len(combos) > 1:
            dest = dest / f"{rule}_t{threshold:g}"
        print(f"Oracle rule={rule} threshold={threshold:g} out={dest}")
        aggregates.append(run_oracle(pairs, args.config, threshold, rule, dest))
    print(json.dumps(aggregates if len(aggregates) > 1 else aggregates[0], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
