"""Build the authors and strict system prompts from EEG info and tool schemas."""
import json
from functools import lru_cache
from pathlib import Path

from planner_adapt import dump_tool_schemas, planner_tool_rules

AUTHORS_TOOL_SCHEMAS_PATH = Path(__file__).resolve().parent / "config" / "protocols" / "authors_tool_schemas.json"


@lru_cache(maxsize=1)
def _authors_schema_index() -> dict:
    with AUTHORS_TOOL_SCHEMAS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {item["function"]["name"]: item for item in payload["schemas"]}


def authors_tool_schemas(tool_meta) -> list:
    """Swap each live schema for the text the authors' commit registered, keeping order."""
    index = _authors_schema_index()
    names = [item["function"]["name"] for item in tool_meta]
    missing = [name for name in names if name not in index]
    if missing:
        raise KeyError(f"{AUTHORS_TOOL_SCHEMAS_PATH.name} has no schema for {missing}")
    return [index[name] for name in names]


def authors_system_prompt(base_info, tool_meta, knowledge):
    """System prompt from the authors' TUEV eval (git acd2e6a), including its wording."""
    prompt = f"""
    You are an assistant specialized in EEG interpretation. Combine the provided EEG record with prior medical knowledge and available analysis tools to answer patient questions.

    <EEG Information> {base_info} </EEG Information> 
    <EEG Prior Knowledge> {knowledge} </EEG Prior Knowledge> 
    <Tools> {tool_meta} </Tools>
    
    When selecting tools, prioritize minimizing the number of calls. Use the most cost-effective tool that meets the requirements, even if it is not the finest in granularity.
    To call a tool, use exactly this format (zero or more times as needed):
    <FUNCTION> tool_name
    <ARGS> {{ "arg1": value1, "arg2": value2, ... }}
    The tool will return:
    <RETURN> tool output
    If a requested time interval is longer than you selected tool allows, split the interval into consecutive valid tool calls that satisfy that tool's time limit. Example: If the user asks to inspect 15-32 seconds and the selected tool allows at most 10 seconds, call the tool on 15-25 and 25-32, not 15-32.
    Catious: All tool calls that reference time must use indices within this recording\u2019s duration of {base_info['data_duration']} seconds.
    """
    return prompt


def strict_system_prompt(base_info, tool_meta, knowledge):
    """Local 0.50-rule prompt. The same rules are appended for every planner."""
    duration = base_info.get("data_duration", "unknown")
    prompt = f"""
You are an assistant specialized in EEG interpretation. Combine the provided EEG record with prior medical knowledge and available analysis tools to answer patient questions.

<EEG Information> {base_info} </EEG Information>
<EEG Prior Knowledge> {knowledge} </EEG Prior Knowledge>
<Tools>
{dump_tool_schemas(tool_meta)}
</Tools>

When selecting tools, prioritize minimizing the number of calls. Use the most cost-effective tool that meets the requirements, even if it is not the finest in granularity.
To call a tool, use exactly this format (zero or more times as needed):
<FUNCTION> tool_name
<ARGS> {{"name": ["FP1-F7", "F7-T3"], "start": 0, "end": 10}}
The tool will return:
<RETURN> tool output
If a requested time interval is longer than the selected tool allows, split the interval into consecutive valid tool calls that satisfy that tool's time limit. Example: if the user asks to inspect 15-32 seconds and the selected tool allows at most 10 seconds, call the tool on 15-25 and 25-32, not 15-32.
Caution: all tool calls that reference time must use indices within this recording's duration of {duration} seconds.
{planner_tool_rules()}
"""
    return prompt


def getSystemPrompt(base_info, tool_meta, knowledge, report_template, planner_model=None, prompt_mode="authors"):
    del planner_model, report_template
    if prompt_mode == "strict":
        return strict_system_prompt(base_info, tool_meta, knowledge)
    if prompt_mode != "authors":
        raise ValueError(f"Unknown prompt_mode {prompt_mode!r}. Choose 'authors' or 'strict'.")
    return authors_system_prompt(base_info, authors_tool_schemas(tool_meta), knowledge)
