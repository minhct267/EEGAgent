"""Shared harness flags for tool results, stop strings, retries, and the final turn."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

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

DISABLED_THINKING = frozenset({"", "none", "off", "disabled"})


@dataclass(frozen=True)
class HarnessConfig:
    """Loop behavior shared by every planner under one protocol."""

    name: str
    tool_result_role: str
    stop: tuple
    empty_retry: bool
    force_final: bool
    # "authors": original regex and json.loads; bad calls are dropped with no feedback.
    # "repaired": lenient ARGS parsing; bad or unknown calls come back as tool errors.
    tool_calls: str


HARNESS_AUTHORS_V2 = HarnessConfig(
    name="authors_v2",
    tool_result_role="user",
    stop=("<RETURN>",),
    empty_retry=True,
    force_final=True,
    tool_calls="repaired",
)
HARNESS_AUTHORS_V1 = HarnessConfig(
    name="authors_v1",
    tool_result_role="assistant",
    stop=(),
    empty_retry=False,
    force_final=False,
    tool_calls="authors",
)
HARNESSES = {
    "authors_v2": HARNESS_AUTHORS_V2,
    "v2": HARNESS_AUTHORS_V2,
    "authors_v1": HARNESS_AUTHORS_V1,
    "v1": HARNESS_AUTHORS_V1,
}


def resolve_harness(name: str | None) -> HarnessConfig:
    """Return the named harness. authors_v2 is the default."""
    key = (name or "authors_v2").strip()
    if key not in HARNESSES:
        known = ", ".join(sorted({"authors_v1", "authors_v2"}))
        raise ValueError(f"Unknown harness {name!r}. Choose from: {known}.")
    return HARNESSES[key]


def thinking_is_disabled(reasoning_effort: str | None) -> bool:
    """True when reasoning_effort is empty or one of none/off/disabled."""
    return (reasoning_effort or "").strip().lower() in DISABLED_THINKING


def preserve_planner_reasoning(model: str | None = None, reasoning_effort: str | None = None) -> bool:
    """Keep think blocks in history only when thinking is enabled."""
    del model
    return not thinking_is_disabled(reasoning_effort)


def truncate_at_stop(text: str | None, stops: tuple | list | None) -> tuple[str, bool]:
    """Drop text at the first stop marker so a simulated tool return is not stored."""
    body = text or ""
    if not body or not stops:
        return body, False
    cut = None
    for stop in stops:
        if not stop:
            continue
        index = body.find(stop)
        if index != -1 and (cut is None or index < cut):
            cut = index
    if cut is None:
        return body, False
    return body[:cut].rstrip(), True


EMPTY_RETRY_MESSAGE = (
    "The previous model turn was empty. Using the tool results already "
    "in this conversation, write the final answer now. If you still need "
    "a tool, emit <FUNCTION>/<ARGS> with valid JSON."
)

FORCE_FINAL_MESSAGE = (
    "Stop calling tools. Using the tool results already in this conversation, "
    "write the final answer now. For each detected seizure, return exactly one line:\n"
    "(channel_name, time_start, time_end)\n"
    "If there are no seizures, write exactly: No events found\n"
    "Do not include any extra text."
)


def dump_tool_schemas(tool_meta) -> str:
    """Pretty-print tool schemas for the strict system prompt."""
    return json.dumps(tool_meta, ensure_ascii=False, indent=2)


def filter_tool_schemas(schemas: list, names: tuple[str, ...] | list[str] | None) -> list:
    """Keep only the named tools. None keeps every registered schema."""
    if not names:
        return schemas
    allowed = set(names)
    return [item for item in schemas if item.get("function", {}).get("name") in allowed]


def planner_tool_rules(model: str | None = None) -> str:
    """Strict-prompt rules. The same text is used for every planner."""
    del model
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
- Call tools only with the <FUNCTION> / <ARGS> XML format shown above. Do not emit OpenAI/native tool_calls.
"""
    return shared
