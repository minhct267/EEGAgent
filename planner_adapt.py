"""Planner prompt and tool-result adaptations for MiniMax and Qwen3.8."""

from __future__ import annotations

import json
import re

DISCHARGE_TOOL_NAMES = (
    "normalAbnormalModel",
    "slowSeizBckgModel_TenSeconds",
    "seizureNormalModel_OneSecond",
    "seizureArtiBckgModel_OneSecond",
    "eyemMuscleModel_OneSecond",
    "compute_amplitude",
    "compute_psd",
    "compute_symmetry",
    "reflectData",
)

TOOL_RESULT_NOTES = {
    "eyemMuscleModel_OneSecond": (
        "Eye-vs-Muscle only. The two class probabilities always sum to about 1.0. "
        "There is no clean or no-artifact class. High Muscle artifact must not be used "
        "to discard seizure or epileptiform findings from other classifiers."
    ),
    "seizureArtiBckgModel_OneSecond": (
        "Three-way scores: background vs artifact vs seizure. "
        "This is the tool that can say a high seizure score is artifact-driven."
    ),
    "seizureNormalModel_OneSecond": (
        "Seizure vs non-seizure only. These scores do not identify artifact type. "
        "If you need artifact vs seizure, use seizureArtiBckgModel_OneSecond."
    ),
    "slowSeizBckgModel_TenSeconds": (
        "Coarse 10-second montage scores (background vs slow vs seizure). "
        "Use 1-second tools next to localize channel and time."
    ),
    "normalAbnormalModel": (
        "Whole-record normal vs abnormal. Abnormal does not by itself localize discharges."
    ),
}

MINIMAX_NAME_RE = re.compile(r"minimax", re.IGNORECASE)
QWEN38_NAME_RE = re.compile(r"qwen3\.8", re.IGNORECASE)
DISABLED_THINKING = frozenset({"", "none", "off", "disabled"})


def is_minimax_planner(model: str | None) -> bool:
    return bool(model and MINIMAX_NAME_RE.search(model))


def is_qwen38_planner(model: str | None) -> bool:
    return bool(model and QWEN38_NAME_RE.search(model))


def thinking_is_disabled(reasoning_effort: str | None) -> bool:
    return (reasoning_effort or "").strip().lower() in DISABLED_THINKING


def preserve_planner_reasoning(model: str | None, reasoning_effort: str | None = None) -> bool:
    """Keep think blocks in history only when the planner is actually thinking."""
    if is_minimax_planner(model):
        return True
    if is_qwen38_planner(model):
        return not thinking_is_disabled(reasoning_effort)
    return False


def tool_results_as_user(model: str | None) -> bool:
    """Show MiniMax tool output as a new user turn so it is not treated as self-talk."""
    return is_minimax_planner(model)


def continue_after_tools(model: str | None) -> bool:
    """Qwen3.8 often emits an empty turn after tool merge; ask it to continue."""
    return is_qwen38_planner(model)


CONTINUE_AFTER_TOOLS = (
    "Continue from the tool results already in this conversation. "
    "If you still need to inspect other seconds in the asked interval, emit more "
    "<FUNCTION>/<ARGS> calls with valid JSON. "
    "Report only channels where a 1-second tool has seiz as the top class and seiz >= 0.50. "
    "Merge consecutive high-seiz seconds on the same channel into one tuple. "
    "If none meet that threshold, write exactly: No events found"
)


def dump_tool_schemas(tool_meta) -> str:
    return json.dumps(tool_meta, ensure_ascii=False, indent=2)


def filter_tool_schemas(schemas: list, names: tuple[str, ...] | list[str] | None) -> list:
    if not names:
        return schemas
    allowed = set(names)
    return [item for item in schemas if item.get("function", {}).get("name") in allowed]


def planner_tool_rules(model: str | None) -> str:
    shared = """
Tool-result rules (must follow):
- eyemMuscleModel_OneSecond only classifies artifact TYPE: Eye movement vs Muscle. It is not a signal-quality check and has no clean class. Never use it to veto seizure or discharge detections.
- seizureArtiBckgModel_OneSecond is the fine-grained tool for Background vs Artifact vs Seizure.
- Typical discharge workflow: optional normalAbnormalModel, then slowSeizBckgModel_TenSeconds on the asked interval, then seizureNormalModel_OneSecond and/or seizureArtiBckgModel_OneSecond on promising seconds and channels.
- Use only channel names listed in available_channel. Do not invent channels.
- Do not call sleepStageModel or healthMDDModel unless the user asked about sleep staging or depression.
- When the user asks for events as (channel_name, time_start, time_end), the final answer must list those tuples with parentheses, one event per line. Example: (FP1-F7, 0.0, 3.0)
- Report an event only when a 1-second tool shows seiz as the highest class and seiz >= 0.50. Do not report from coarse 10-second scores alone.
- If consecutive seconds on the same channel stay at or above that threshold, emit one merged tuple covering the full run.
- If no 1-second window meets the threshold, write exactly: No events found
- Do not invent events just to fill the tuple format.
- When localizing an interval [start, end], inspect the beginning of that interval first (including the first 10 seconds). If you subsample 1-second tools, cover beginning, middle, and end, plus any coarse 10-second window with high seiz.
- Every <ARGS> object must be valid JSON: double-quoted keys and strings, no Python single quotes, no unquoted values.
"""
    if is_minimax_planner(model):
        return (
            shared
            + """
MiniMax-M3 format:
- Call tools only with the <FUNCTION> / <ARGS> XML format shown above. Do not use OpenAI/native tool_calls.
- After tool results arrive, update the conclusion from the classifier scores. Do not keep a prior "artifact contamination" narrative if seizure or discharge scores remain high.
"""
        )
    if is_qwen38_planner(model):
        return (
            shared
            + """
Qwen3.8 format:
- Call tools only with the <FUNCTION> / <ARGS> XML format shown above.
- Every <ARGS> object must be valid JSON with double-quoted keys and strings.
- Do not emit OpenAI/native tool_calls.
- Cover every integer second of the asked interval with 1-second tools before the final answer.
- Prefer writing only tuple lines, or the exact sentence No events found.
"""
        )
    return shared
