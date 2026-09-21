"""Sanity-check MiniMax discharge-tool prompt wiring."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from planner_adapt import DISCHARGE_TOOL_NAMES, filter_tool_schemas, planner_tool_rules
from prompt import getSystemPrompt
from tools import function_register


def main() -> None:
    schemas = filter_tool_schemas(function_register.export_tool_schemas(), DISCHARGE_TOOL_NAMES)
    names = [item["function"]["name"] for item in schemas]
    print("tools", names)
    if "sleepStageModel" in names or "healthMDDModel" in names:
        raise SystemExit("Sleep/MDD tools should not be in the discharge subset.")
    if set(DISCHARGE_TOOL_NAMES) - set(names):
        raise SystemExit(f"Missing discharge tools: {set(DISCHARGE_TOOL_NAMES) - set(names)}")

    eye = next(item for item in schemas if item["function"]["name"] == "eyemMuscleModel_OneSecond")
    if "NOT a clean-vs-artifact" not in eye["function"]["description"]:
        raise SystemExit("eyemMuscle description is missing the clean-vs-artifact warning.")

    rules = planner_tool_rules("minimax-m3")
    if "Never use it to veto" not in rules or "MiniMax-M3 format" not in rules:
        raise SystemExit("MiniMax planner rules are incomplete.")

    prompt = getSystemPrompt(
        {"data_duration": 300, "available_channel": ["FP1-F7"]},
        schemas,
        {"k": 1},
        {},
        planner_model="minimax-m3",
    )
    if '"name": ["FP1-F7", "F7-T3"]' not in prompt:
        raise SystemExit("Prompt example is not valid JSON.")
    if "Eye movement vs Muscle" not in prompt:
        raise SystemExit("Prompt is missing Eye-vs-Muscle rule.")
    print(f"prompt_ok chars={len(prompt)}")


if __name__ == "__main__":
    main()
