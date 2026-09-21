"""Parser checks for MiniMax/Qwen tool-call text."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from TUEV_eval import parse_events  # noqa: E402
from utils.parseCalling import extract_tool_calls, parse_tool_args  # noqa: E402


def expect(condition: bool, detail: str) -> None:
    if not condition:
        raise SystemExit(f"[FAIL] {detail}")
    print(f"[PASS] {detail}")


def main() -> None:
    single = extract_tool_calls(
        "<FUNCTION> seizureNormalModel_OneSecond\n"
        "<ARGS> {'name': ['FP1-F7', 'F7-T3'], 'start': 0, 'end': 10}\n"
    )
    expect(
        single == [{"name": "seizureNormalModel_OneSecond", "args": {"name": ["FP1-F7", "F7-T3"], "start": 0, "end": 10}}],
        "Python single-quoted ARGS",
    )

    unquoted = extract_tool_calls(
        "<FUNCTION> compute_amplitude\n"
        "<ARGS> {name: [\"FP1-F7\"], start: 0, end: 10}\n"
    )
    expect(
        unquoted == [{"name": "compute_amplitude", "args": {"name": ["FP1-F7"], "start": 0, "end": 10}}],
        "Unquoted JSON keys",
    )

    fenced = extract_tool_calls(
        "<FUNCTION> eyemMuscleModel_OneSecond</FUNCTION>\n"
        "<ARGS>\n```json\n{\"name\": [\"FP1-F7\"], \"start\": 0, \"end\": 5}\n```\n"
    )
    expect(
        fenced == [{"name": "eyemMuscleModel_OneSecond", "args": {"name": ["FP1-F7"], "start": 0, "end": 5}}],
        "Closing tag plus markdown fence",
    )

    think = extract_tool_calls(
        "<think>I will call a tool.\n"
        "<FUNCTION> slowSeizBckgModel_TenSeconds\n"
        "<ARGS> {\"start\": 0, \"end\": 60}\n"
        "</think>\nNo discharges."
    )
    expect(
        think == [{"name": "slowSeizBckgModel_TenSeconds", "args": {"start": 0, "end": 60}}],
        "Tool call inside think tags",
    )

    invoke = extract_tool_calls(
        '<invoke name="seizureArtiBckgModel_OneSecond">'
        '<parameter name="name">["FP2-F8"]</parameter>'
        '<parameter name="start">30</parameter>'
        '<parameter name="end">35</parameter>'
        "</invoke>"
    )
    expect(
        invoke == [{"name": "seizureArtiBckgModel_OneSecond", "args": {"name": ["FP2-F8"], "start": 30, "end": 35}}],
        "MiniMax invoke XML",
    )

    parsed = parse_tool_args('{"name": ["FP1-F7"], "start": 0, "end": 10,}')
    expect(parsed["end"] == 10, "Trailing comma in JSON object")

    paren_events = parse_events("(FP1-F7, 0.0, 3.0)\n(FP2-F8, 2.6, 5.0)")
    expect(
        paren_events == [
            {"channel": "FP1-F7", "start_time": 0.0, "end_time": 3.0},
            {"channel": "FP2-F8", "start_time": 2.6, "end_time": 5.0},
        ],
        "Parenthesized TUEV events",
    )
    line_events = parse_events("- FP1-F7, 25, 26\n- FP2-F8, 5, 6")
    expect(
        line_events == [
            {"channel": "FP1-F7", "start_time": 25.0, "end_time": 26.0},
            {"channel": "FP2-F8", "start_time": 5.0, "end_time": 6.0},
        ],
        "MiniMax dashed TUEV events",
    )
    print("All parser checks passed.")


if __name__ == "__main__":
    main()
