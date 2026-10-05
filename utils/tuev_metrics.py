"""Event-level TUEV scoring: merge reports, count ground-truth hits, and classify unmatched episodes."""

import json
import math
import random
from pathlib import Path

COUNT_KEYS = [
    "files",
    "total_gt",
    "total_reports",
    "hits",
    "misses",
    "strict_unmatched_reports",
    "explicit_negative_reports",
    "unverified_reports",
]

V2_COUNT_KEYS = [
    "iou_hits",
    "positive_overlapping_reports",
    "zero_gt_files",
    "zero_gt_reports",
    "spsw_gt",
    "spsw_hits",
    "spsw_iou_hits",
    "gped_gt",
    "gped_hits",
    "gped_iou_hits",
    "pled_gt",
    "pled_hits",
    "pled_iou_hits",
]

CLASS_FIELDS = {1: "spsw", 2: "gped", 3: "pled"}
IOU_HIT_THRESHOLD = 0.7


def write_json(path, data):
    """Write indented JSON, creating parent directories when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_summary(path):
    """Load one per-file summary.json."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def aggregate_summaries(summaries):
    """Sum per-file counts and recompute the pooled rates."""
    totals = {key: 0 for key in COUNT_KEYS}
    thresholds = set()
    models = set()

    for summary in summaries:
        for key in COUNT_KEYS:
            totals[key] += int(summary.get(key, 0))
        if "threshold" in summary:
            thresholds.add(summary["threshold"])
        if "model" in summary:
            models.add(summary["model"])

    total_gt = totals["total_gt"]
    total_reports = totals["total_reports"]
    result = {
        **totals,
        "hit_rate": totals["hits"] / total_gt if total_gt else math.nan,
        "strict_unmatched_report_rate": (
            totals["strict_unmatched_reports"] / total_reports if total_reports else math.nan
        ),
        "explicit_negative_report_rate": (
            totals["explicit_negative_reports"] / total_reports if total_reports else math.nan
        ),
        "unverified_report_rate": (
            totals["unverified_reports"] / total_reports if total_reports else math.nan
        ),
    }
    if thresholds:
        result["thresholds"] = sorted(thresholds)
    if models:
        result["models"] = sorted(models)
    result.update(aggregate_v2(summaries))
    return result


def aggregate_v2(summaries):
    """Add IoU, per-class recall, event F1, and file-level bootstrap intervals."""
    if not summaries or not any("iou_hits" in summary for summary in summaries):
        return {}
    totals = {key: 0 for key in V2_COUNT_KEYS}
    for summary in summaries:
        for key in V2_COUNT_KEYS:
            totals[key] += int(summary.get(key, 0))
    total_gt = sum(int(summary.get("total_gt", 0)) for summary in summaries)
    total_reports = sum(int(summary.get("total_reports", 0)) for summary in summaries)
    hits = sum(int(summary.get("hits", 0)) for summary in summaries)
    hit_rate = _finite_rate(hits, total_gt)
    event_precision = _finite_rate(totals["positive_overlapping_reports"], total_reports)
    v2 = {
        **totals,
        "iou_hit_rate": _finite_rate(totals["iou_hits"], total_gt),
        "event_precision": event_precision,
        "event_f1": _f1(event_precision, hit_rate),
        "spsw_recall": _finite_rate(totals["spsw_hits"], totals["spsw_gt"]),
        "gped_recall": _finite_rate(totals["gped_hits"], totals["gped_gt"]),
        "pled_recall": _finite_rate(totals["pled_hits"], totals["pled_gt"]),
        "spsw_iou_recall": _finite_rate(totals["spsw_iou_hits"], totals["spsw_gt"]),
        "gped_iou_recall": _finite_rate(totals["gped_iou_hits"], totals["gped_gt"]),
        "pled_iou_recall": _finite_rate(totals["pled_iou_hits"], totals["pled_gt"]),
        "hit_rate_bootstrap_95": bootstrap_file_rate(
            [int(summary.get("hits", 0)) for summary in summaries],
            [int(summary.get("total_gt", 0)) for summary in summaries],
        ),
        "iou_hit_rate_bootstrap_95": bootstrap_file_rate(
            [int(summary.get("iou_hits", 0)) for summary in summaries],
            [int(summary.get("total_gt", 0)) for summary in summaries],
        ),
    }
    return v2


def _finite_rate(numerator, denominator):
    """Return numerator/denominator, or NaN when the denominator is zero."""
    if not denominator:
        return float("nan")
    return numerator / denominator


def _f1(precision, recall):
    """Harmonic mean of precision and recall. NaN in either input stays NaN."""
    if precision != precision or recall != recall or precision + recall == 0:
        return float("nan") if precision != precision or recall != recall else 0.0
    return 2 * precision * recall / (precision + recall)


def _percentile(sorted_values, fraction):
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_file_rate(numerators, denominators, draws=1000, seed=0):
    """File-level bootstrap of a pooled rate. Files are resampled with replacement."""
    count = len(numerators)
    if count == 0:
        return None
    rng = random.Random(seed)
    rates = []
    for _ in range(draws):
        numerator = 0
        denominator = 0
        for _pick in range(count):
            index = rng.randrange(count)
            numerator += numerators[index]
            denominator += denominators[index]
        if denominator > 0:
            rates.append(numerator / denominator)
    if not rates:
        return None
    rates.sort()
    return {
        "draws": len(rates),
        "seed": seed,
        "low": _percentile(rates, 0.025),
        "high": _percentile(rates, 0.975),
        "mean": sum(rates) / len(rates),
    }


def best_report_iou(gt, reports):
    """Highest IoU between one ground-truth event and any same-channel report."""
    gt_length = gt["end"] - gt["start"]
    best = 0.0
    for report in reports:
        if report["channel"] != gt.get("channel_name"):
            continue
        overlap = calculate_overlap(
            [gt["start"], gt["end"]],
            [report["start_time"], report["end_time"]],
        )
        union = gt_length + (report["end_time"] - report["start_time"]) - overlap
        if union > 0:
            best = max(best, overlap / union)
    return best


def calculate_overlap(box_a, box_b):
    """Length of the overlap between two [start, end] intervals."""
    inter_start = max(box_a[0], box_b[0])
    inter_end = min(box_a[1], box_b[1])
    return max(0, inter_end - inter_start)


def merge_intervals(intervals):
    """Merge overlapping [start, end] intervals and drop empty ones."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def interval_total_length(intervals):
    """Sum the lengths of intervals that are already non-overlapping."""
    return sum(end - start for start, end in intervals)


def merge_predictions(predictions, gap_threshold=1.0):
    """Merge same-channel reports whose gap is within the threshold."""
    grouped = {}
    for pred in predictions:
        grouped.setdefault(pred["channel"], []).append(pred)

    merged = []
    for channel, group in grouped.items():
        current = None
        for pred in sorted(group, key=lambda item: item["start_time"]):
            if current is None:
                current = pred.copy()
            elif pred["start_time"] <= current["end_time"] + gap_threshold:
                current["end_time"] = max(current["end_time"], pred["end_time"])
            else:
                merged.append(current)
                current = pred.copy()
        if current is not None:
            merged.append(current)
    return sorted(merged, key=lambda item: (item["start_time"], item["channel"]))


def score_predictions(gt_events, negative_events, raw_predictions, threshold=0.7, gap_threshold=1.0):
    """Merge nearby same-channel reports. A ground-truth event hits when coverage reaches the threshold."""
    gt_events = list(gt_events or [])
    negative_events = list(negative_events or [])
    raw_predictions = list(raw_predictions or [])
    report_episodes = merge_predictions(raw_predictions, gap_threshold=gap_threshold)
    detected_gt = set()

    for gt_index, gt in enumerate(gt_events):
        gt_length = gt["end"] - gt["start"]
        overlap_intervals = []
        for report in report_episodes:
            if report["channel"] != gt["channel_name"]:
                continue
            overlap = calculate_overlap([gt["start"], gt["end"]], [report["start_time"], report["end_time"]])
            if overlap <= 0:
                continue
            overlap_intervals.append([
                max(gt["start"], report["start_time"]),
                min(gt["end"], report["end_time"]),
            ])
        covered_length = interval_total_length(merge_intervals(overlap_intervals))
        if gt_length > 0 and covered_length / gt_length >= threshold:
            detected_gt.add(gt_index)

    scored_report_episodes = []
    positive_overlapping_reports = 0
    hit_contributing_reports = 0
    strict_unmatched_reports = 0
    explicit_negative_reports = 0
    unverified_reports = 0

    for report in report_episodes:
        report_box = [report["start_time"], report["end_time"]]
        overlapping_gt_indices = [
            gt_index
            for gt_index, gt in enumerate(gt_events)
            if report["channel"] == gt["channel_name"]
            and calculate_overlap([gt["start"], gt["end"]], report_box) > 0
        ]
        has_positive_overlap = bool(overlapping_gt_indices)
        has_negative_overlap = any(
            report["channel"] == neg["channel_name"]
            and calculate_overlap([neg["start"], neg["end"]], report_box) > 0
            for neg in negative_events
        )

        scored_report = report.copy()
        scored_report["overlapping_gt_indices"] = overlapping_gt_indices
        scored_report["is_false_positive"] = not has_positive_overlap
        scored_report["contributes_to_hit"] = any(gt_index in detected_gt for gt_index in overlapping_gt_indices)
        scored_report_episodes.append(scored_report)

        if has_positive_overlap:
            positive_overlapping_reports += 1
            if scored_report["contributes_to_hit"]:
                hit_contributing_reports += 1
            continue

        # No positive overlap: unmatched, then split into explicit-negative vs unverified.
        strict_unmatched_reports += 1
        if has_negative_overlap:
            explicit_negative_reports += 1
        else:
            unverified_reports += 1

    total_gt = len(gt_events)
    total_raw_preds = len(raw_predictions)
    total_preds = len(report_episodes)
    hits = len(detected_gt)
    class_counts = {label: {"gt": 0, "hits": 0, "iou_hits": 0} for label in CLASS_FIELDS.values()}
    iou_hit_indices = set()
    for gt_index, gt in enumerate(gt_events):
        label = CLASS_FIELDS.get(int(gt.get("class", -1)))
        if label:
            class_counts[label]["gt"] += 1
            if gt_index in detected_gt:
                class_counts[label]["hits"] += 1
        if best_report_iou(gt, report_episodes) > IOU_HIT_THRESHOLD:
            iou_hit_indices.add(gt_index)
            if label:
                class_counts[label]["iou_hits"] += 1
    iou_hits = len(iou_hit_indices)
    hit_rate = hits / total_gt if total_gt else float("nan")
    event_precision = positive_overlapping_reports / total_preds if total_preds else float("nan")

    metrics = {
        "total_gt": total_gt,
        "total_raw_preds": total_raw_preds,
        "total_preds": total_preds,
        "hits": hits,
        "misses": total_gt - hits,
        "positive_overlapping_reports": positive_overlapping_reports,
        "hit_contributing_reports": hit_contributing_reports,
        "no_positive_overlap_predictions": strict_unmatched_reports,
        "false_positives": strict_unmatched_reports,
        "strict_unmatched_reports": strict_unmatched_reports,
        "explicit_negative_reports": explicit_negative_reports,
        "unverified_reports": unverified_reports,
        "hit_rate": hits / total_gt if total_gt else float("nan"),
        "positive_overlap_report_rate": positive_overlapping_reports / total_preds if total_preds else float("nan"),
        "hit_contributing_report_rate": hit_contributing_reports / total_preds if total_preds else float("nan"),
        "false_alarm_rate": strict_unmatched_reports / total_preds if total_preds else float("nan"),
        "strict_unmatched_report_rate": strict_unmatched_reports / total_preds if total_preds else float("nan"),
        "explicit_negative_report_rate": explicit_negative_reports / total_preds if total_preds else float("nan"),
        "unverified_report_rate": unverified_reports / total_preds if total_preds else float("nan"),
        "iou_hits": iou_hits,
        "iou_hit_rate": iou_hits / total_gt if total_gt else float("nan"),
        "event_precision": event_precision,
        "event_f1": _f1(event_precision, hit_rate),
        "zero_gt_files": 0 if total_gt else 1,
        "zero_gt_reports": 0 if total_gt else total_preds,
    }
    for label, counts in class_counts.items():
        metrics[f"{label}_gt"] = counts["gt"]
        metrics[f"{label}_hits"] = counts["hits"]
        metrics[f"{label}_iou_hits"] = counts["iou_hits"]
        metrics[f"{label}_recall"] = _finite_rate(counts["hits"], counts["gt"])
        metrics[f"{label}_iou_recall"] = _finite_rate(counts["iou_hits"], counts["gt"])
    return metrics, report_episodes, scored_report_episodes


def summarize_metrics(metrics, model):
    """Keep the counts that aggregate_summaries adds across files."""
    return {
        "files": 1,
        "model": model,
        "total_gt": metrics["total_gt"],
        "total_reports": metrics["total_preds"],
        "hit_rate": metrics["hit_rate"],
        "strict_unmatched_report_rate": metrics["strict_unmatched_report_rate"],
        "explicit_negative_report_rate": metrics["explicit_negative_report_rate"],
        "unverified_report_rate": metrics["unverified_report_rate"],
        "hits": metrics["hits"],
        "misses": metrics["misses"],
        "strict_unmatched_reports": metrics["strict_unmatched_reports"],
        "explicit_negative_reports": metrics["explicit_negative_reports"],
        "unverified_reports": metrics["unverified_reports"],
        "iou_hits": metrics.get("iou_hits", 0),
        "positive_overlapping_reports": metrics.get("positive_overlapping_reports", 0),
        "zero_gt_files": metrics.get("zero_gt_files", 0),
        "zero_gt_reports": metrics.get("zero_gt_reports", 0),
        "spsw_gt": metrics.get("spsw_gt", 0),
        "spsw_hits": metrics.get("spsw_hits", 0),
        "spsw_iou_hits": metrics.get("spsw_iou_hits", 0),
        "gped_gt": metrics.get("gped_gt", 0),
        "gped_hits": metrics.get("gped_hits", 0),
        "gped_iou_hits": metrics.get("gped_iou_hits", 0),
        "pled_gt": metrics.get("pled_gt", 0),
        "pled_hits": metrics.get("pled_hits", 0),
        "pled_iou_hits": metrics.get("pled_iou_hits", 0),
    }


def write_file_outputs(out_dir, stem, edf_path, rec_path, model, gt_events, negative_events, raw_predictions,
                       metrics, scored_report_episodes, raw_logs, report_overlap_threshold=0.7):
    """Write one recording's raw log, transcript, predictions, and metrics."""
    file_out_dir = Path(out_dir) / stem
    messages_dir = file_out_dir / "messages"
    file_out_dir.mkdir(parents=True, exist_ok=True)
    messages_dir.mkdir(parents=True, exist_ok=True)
    messages_file = messages_dir / f"{stem}.messages.json"

    with (file_out_dir / "agent_raw.jsonl").open("w", encoding="utf-8") as raw_f:
        for log in raw_logs:
            raw_log = {key: value for key, value in log.items() if key != "messages"}
            raw_log["messages_file"] = str(messages_file)
            raw_f.write(json.dumps(raw_log, ensure_ascii=False, default=str) + "\n")

    write_json(messages_file, {
        "stem": stem,
        "edf": edf_path,
        "rec": rec_path,
        "model": model,
        "candidates": [
            {
                "pair_index": log["pair_index"],
                "stem": log["stem"],
                "edf": log["edf"],
                "rec": log["rec"],
                "candidate_index": log["candidate_index"],
                "window": log["window"],
                "question": log["question"],
                "model": log["model"],
                "messages": log["messages"],
                "parsed_events": log["parsed_events"],
                "invalid_events": log.get("invalid_events", []),
                "format_status": log.get("format_status"),
                "error": log["error"],
            }
            for log in raw_logs
        ],
    })

    with (file_out_dir / "agent_predictions.jsonl").open("w", encoding="utf-8") as pred_f:
        pred_f.write(json.dumps({
            "stem": stem,
            "edf": edf_path,
            "rec": rec_path,
            "gt_events": gt_events,
            "negative_events": negative_events,
            "predictions": raw_predictions,
            "scored_report_episodes": scored_report_episodes,
            "metrics": {**metrics, "stem": stem, "edf": edf_path, "rec": rec_path},
            "messages_file": str(messages_file),
        }, ensure_ascii=False, default=str) + "\n")

    write_json(file_out_dir / "metrics.json", {
        "aggregate": {
            "files": 1,
            "model": model,
            "report_overlap_threshold": report_overlap_threshold,
            "gt_policy": "merged_positive_events",
            "prediction_policy": "merged_report_episodes",
            **metrics,
        },
        "per_file": [{**metrics, "stem": stem, "edf": edf_path, "rec": rec_path}],
    })
    write_json(file_out_dir / "summary.json", summarize_metrics(metrics, model))


def write_global_outputs(out_dir):
    """Pool every per-file summary.json under out_dir into the aggregate files."""
    root_out_dir = Path(out_dir)
    summary_files = sorted(root_out_dir.glob("*/summary.json"))
    summaries = [load_summary(path) for path in summary_files]
    if summaries:
        aggregate = aggregate_summaries(summaries)
        aggregate["summary_files"] = [str(path) for path in summary_files]
    else:
        aggregate = {
            "files": 0,
            "total_gt": 0,
            "total_reports": 0,
            "hits": 0,
            "misses": 0,
            "strict_unmatched_reports": 0,
            "explicit_negative_reports": 0,
            "unverified_reports": 0,
            "hit_rate": float("nan"),
            "strict_unmatched_report_rate": float("nan"),
            "explicit_negative_report_rate": float("nan"),
            "unverified_report_rate": float("nan"),
            "summary_files": [],
        }
    write_json(root_out_dir / "aggregate_metrics.json", {"aggregate": aggregate, "summaries": summaries})
    write_json(root_out_dir / "aggregate_summary.json", aggregate)
    return aggregate
