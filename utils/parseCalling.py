import inspect
import json

import regex as re


FUNCTION_HEADER_RE = re.compile(r"<FUNCTION>\s*(\w+)\s+<ARGS>\s*", re.DOTALL)
SINGLE_QUOTED_STRING_RE = re.compile(r"'([^'\\]*)'")


def extract_balanced_object(text, start):
    """Return the substring of a balanced {...} object starting at start, or None."""
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    quote = None
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in ('"', "'"):
            in_string = True
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_tool_args(args_str):
    """Parse tool ARGS, accepting strict JSON or Python-style single quotes."""
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        repaired = SINGLE_QUOTED_STRING_RE.sub(r'"\1"', args_str)
        return json.loads(repaired)


def extract_tool_calls(response):
    if not response:
        return []

    tool_calls = []
    for match in FUNCTION_HEADER_RE.finditer(response):
        tool_name = match.group(1)
        args_str = extract_balanced_object(response, match.end())
        if args_str is None:
            print(f"Parameter parsing failed: no JSON object after <ARGS> for {tool_name}")
            continue
        try:
            args = parse_tool_args(args_str)
        except Exception as exc:
            print(f"Parameter parsing failed: {exc}")
            continue
        tool_calls.append({
            "name": tool_name,
            "args": args,
        })
    return tool_calls


def has_config_parameter(func):
    return "config" in inspect.signature(func).parameters
