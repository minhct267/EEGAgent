"""Re-score saved TUEV predictions with scorer v2, and re-parse raw answers when those logs exist."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tuev_metrics import aggregate_summaries, score_predictions, summarize_metrics, write_json


def _git_bytes(commit: str, path: str) -> bytes:
    """Read one file from a commit without checking it out."""
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=PROJECT_ROOT)


def _git_text(commit: str, path: str) -> str:
    """Read one UTF-8 file from a commit."""
    return _git_bytes(commit, path).decode("utf-8")


def prediction_paths_from_dir(run_dir: Path) -> list[tuple[str, str]]:
    """Return (stem, file text) for every agent_predictions.jsonl under the run."""
    found = []
    for path in sorted(run_dir.glob("*/agent_predictions.jsonl")):
        found.append((path.parent.name, path.read_text(encoding="utf-8")))
    return found


def prediction_paths_from_git(commit: str, prefix: str) -> list[tuple[str, str]]:
    """Same as prediction_paths_from_dir, reading the files from a commit."""
    listing = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, prefix],
        cwd=PROJECT_ROOT,
        text=True,
    )
    found = []
    for path in listing.splitlines():
        if path.endswith("/agent_predictions.jsonl"):
            found.append((Path(path).parent.name, _git_text(commit, path)))
    return found


def raw_text_for(stem: str, run_dir: Path | None, commit: str | None, prefix: str | None) -> str | None:
    """Load agent_raw.jsonl for one stem from disk or from git."""
    if run_dir is not None:
        path = run_dir / stem / "agent_raw.jsonl"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None
    git_path = f"{prefix.rstrip('/')}/{stem}/agent_raw.jsonl"
    try:
        return _git_text(commit, git_path)
    except subprocess.CalledProcessError:
        return None


def _loads_lines(text: str) -> list[dict]:
    """Parse a JSONL string, skipping blank lines."""
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _stored_predictions(rows: list[dict]) -> tuple[list, list, list, int | None]:
    """Take ground truth, negatives, predictions, and the saved hit count from the last row."""
    if not rows:
        return [], [], [], None
    row = rows[-1]
    stored_hits = None
    metrics = row.get("metrics") or {}
    if "hits" in metrics:
        stored_hits = int(metrics["hits"])
    return (
        row.get("gt_events") or [],
        row.get("negative_events") or [],
        row.get("predictions") or row.get("raw_predictions") or [],
        stored_hits,
    )


def _summary_for(stem: str, gt_events, negative_events, predictions) -> tuple[dict, dict]:
    """Score one file and return the aggregate-ready summary plus the full metrics."""
    metrics, _episodes, _scored = score_predictions(gt_events, negative_events, predictions)
    summary = summarize_metrics(metrics, metrics.get("model"))
    summary["stem"] = stem
    summary["model"] = None
    return summary, metrics


def _reparsed_predictions(raw_text: str) -> tuple[list, list]:
    """Parse stored answers again with the current channel canonicalizer."""
    from TUEV_eval import parse_events_detailed

    predictions = []
    invalid = []
    for record in _loads_lines(raw_text):
        text = record.get("raw_response") or ""
        events, invalid_events = parse_events_detailed(text)
        if not events:
            assistant = ((record.get("result_meta") or {}).get("raw_assistant")) or ""
            if assistant:
                events, invalid_events = parse_events_detailed(assistant)
        predictions.extend(events)
        invalid.extend(invalid_events)
    return predictions, invalid


def rescore(records: list[tuple[str, str]], raw_lookup, stems: set[str] | None) -> dict:
    """Recompute coverage from stored predictions and, when raw logs exist, from re-parsed answers."""
    stored_summaries = []
    reparsed_summaries = []
    mismatches = []
    per_file = []
    reparse_invalid = 0
    for stem, text in records:
        if stems is not None and stem not in stems:
            continue
        rows = _loads_lines(text)
        gt_events, negative_events, predictions, stored_hits = _stored_predictions(rows)
        summary, metrics = _summary_for(stem, gt_events, negative_events, predictions)
        if stored_hits is not None and stored_hits != metrics["hits"]:
            mismatches.append({"stem": stem, "stored_hits": stored_hits, "rescored_hits": metrics["hits"]})
        stored_summaries.append(summary)
        file_row = {
            "stem": stem,
            "total_gt": metrics["total_gt"],
            "hits": metrics["hits"],
            "iou_hits": metrics["iou_hits"],
            "total_reports": metrics["total_preds"],
            "strict_unmatched_reports": metrics["strict_unmatched_reports"],
            "spsw_gt": metrics["spsw_gt"],
            "spsw_hits": metrics["spsw_hits"],
            "gped_gt": metrics["gped_gt"],
            "gped_hits": metrics["gped_hits"],
            "pled_gt": metrics["pled_gt"],
            "pled_hits": metrics["pled_hits"],
            "zero_gt_reports": metrics["zero_gt_reports"],
        }
        raw_text = raw_lookup(stem)
        if raw_text is not None:
            reparsed, invalid = _reparsed_predictions(raw_text)
            reparse_invalid += len(invalid)
            re_summary, re_metrics = _summary_for(stem, gt_events, negative_events, reparsed)
            reparsed_summaries.append(re_summary)
            file_row["reparsed_hits"] = re_metrics["hits"]
            file_row["reparsed_iou_hits"] = re_metrics["iou_hits"]
            file_row["reparsed_reports"] = re_metrics["total_preds"]
            file_row["invalid_channels"] = len(invalid)
        per_file.append(file_row)
    return {
        "files": len(stored_summaries),
        "stored_predictions": aggregate_summaries(stored_summaries) if stored_summaries else {},
        "reparsed": aggregate_summaries(reparsed_summaries) if reparsed_summaries else None,
        "reparsed_invalid_events": reparse_invalid,
        "sanity": {
            "coverage_hits_match_stored": not mismatches,
            "mismatches": mismatches,
        },
        "per_file": per_file,
    }


def _stems_from_git(commit: str, prefix: str) -> set[str]:
    """Stems that have an agent_predictions.jsonl in this commit."""
    return {stem for stem, _text in prediction_paths_from_git(commit, prefix)}


def main() -> None:
    """Re-score a run directory or a git commit and write the JSON report."""
    parser = argparse.ArgumentParser(description="Re-score a saved TUEV run with scorer v2.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory of per-file agent_predictions.jsonl.")
    parser.add_argument("--git", default=None, help="Commit to read, for example acd2e6a.")
    parser.add_argument("--git-prefix", default="runs/tuev_agent", help="Path inside the commit.")
    parser.add_argument("--stems-from-git", default=None, help="Keep only stems present in this commit's run.")
    parser.add_argument("--stems-prefix", default="runs/tuev_agent")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-reparse", action="store_true")
    args = parser.parse_args()
    if bool(args.run_dir) == bool(args.git):
        raise SystemExit("Pass exactly one of --run-dir or --git.")

    if args.run_dir:
        records = prediction_paths_from_dir(args.run_dir)
        source = {"kind": "directory", "path": str(args.run_dir.resolve())}

        def raw_lookup(stem, run_dir=args.run_dir):
            if args.no_reparse:
                return None
            return raw_text_for(stem, run_dir, None, None)
    else:
        records = prediction_paths_from_git(args.git, args.git_prefix)
        source = {"kind": "git", "commit": args.git, "prefix": args.git_prefix}

        def raw_lookup(stem, commit=args.git, prefix=args.git_prefix):
            if args.no_reparse:
                return None
            return raw_text_for(stem, None, commit, prefix)

    stems = None
    if args.stems_from_git:
        stems = _stems_from_git(args.stems_from_git, args.stems_prefix)
        source["subset_commit"] = args.stems_from_git
        source["subset_files"] = len(stems)
    payload = rescore(records, raw_lookup, stems)
    payload["source"] = source
    write_json(args.out, payload)
    stored = payload["stored_predictions"]
    print(
        f"files={payload['files']} hits={stored.get('hits')} gt={stored.get('total_gt')} "
        f"hit_rate={stored.get('hit_rate')} iou_hits={stored.get('iou_hits')} "
        f"iou_hit_rate={stored.get('iou_hit_rate')} "
        f"sanity={payload['sanity']['coverage_hits_match_stored']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
