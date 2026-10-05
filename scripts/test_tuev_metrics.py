"""Scorer v2 checks that do not load the planner or the EEG tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.tuev_metrics import aggregate_summaries, bootstrap_file_rate, score_predictions, summarize_metrics


def expect(condition: bool, detail: str) -> None:
    """Exit on the first failed check and print a PASS line otherwise."""
    if not condition:
        raise SystemExit(f"[FAIL] {detail}")
    print(f"[PASS] {detail}")


def main() -> None:
    """Check coverage hits, IoU, per-class counts, and the file-level bootstrap."""
    gt = [
        {"channel": 0, "channel_name": "FP1-F7", "start": 10.0, "end": 12.0, "class": 1},
        {"channel": 1, "channel_name": "F7-T3", "start": 10.0, "end": 11.0, "class": 2},
    ]
    predictions = [
        {"channel": "FP1-F7", "start_time": 10.0, "end_time": 12.0},
        {"channel": "F7-T3", "start_time": 10.0, "end_time": 20.0},
    ]
    metrics, _reports, _scored = score_predictions(gt, [], predictions)
    expect(metrics["hits"] == 2, "Coverage hits both events")
    expect(metrics["iou_hits"] == 1, "Only the exact interval passes IoU > 0.7")
    expect(metrics["spsw_hits"] == 1 and metrics["gped_hits"] == 1, "Per-class coverage hits")
    expect(metrics["spsw_iou_hits"] == 1 and metrics["gped_iou_hits"] == 0, "Per-class IoU hits")
    expect(metrics["event_precision"] == 1.0, "Both reports overlap a positive event")
    expect(abs(metrics["event_f1"] - 1.0) < 1e-9, "F1 is 1 when precision and recall are 1")

    empty_gt, _reports, _scored = score_predictions([], [{"channel_name": "FP1-F7", "start": 0, "end": 1, "class": 6}], predictions)
    expect(empty_gt["zero_gt_files"] == 1 and empty_gt["zero_gt_reports"] == 2, "Reports on a file with no positive GT")
    expect(empty_gt["hits"] == 0 and empty_gt["total_gt"] == 0, "Zero GT does not invent hits")

    partial = score_predictions(
        [{"channel_name": "FP1-F7", "start": 0.0, "end": 10.0, "class": 3}],
        [],
        [{"channel": "FP1-F7", "start_time": 0.0, "end_time": 7.0}],
    )[0]
    expect(partial["hits"] == 1 and partial["iou_hits"] == 0, "70% coverage hits while IoU stays at 0.7 and does not exceed it")

    summaries = [summarize_metrics(metrics, "m"), summarize_metrics(empty_gt, "m")]
    aggregate = aggregate_summaries(summaries)
    expect(aggregate["hits"] == 2 and aggregate["iou_hits"] == 1, "Aggregate keeps coverage and IoU counts")
    expect(aggregate["zero_gt_files"] == 1 and aggregate["zero_gt_reports"] == 2, "Aggregate counts zero-GT reports")
    interval = bootstrap_file_rate([2, 0], [2, 0], draws=200, seed=0)
    expect(interval["low"] == 1.0 and interval["high"] == 1.0, "Bootstrap is 1 when every GT file is a full hit")
    print("All scorer checks passed.")


if __name__ == "__main__":
    main()
