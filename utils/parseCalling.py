import ast
import inspect
import json

import regex as re


FUNCTION_HEADER_RE = re.compile(
    r"<FUNCTION>\s*(\w+)\s*(?:</FUNCTION>)?\s*<ARGS>\s*(?:</ARGS>)?\s*",
    re.DOTALL | re.IGNORECASE,
)
INVOKE_RE = re.compile(
    r"<invoke\s+name=[\"'](\w+)[\"']\s*>(.*?)</invoke>",
    re.DOTALL | re.IGNORECASE,
)
PARAM_RE = re.compile(
    r"<parameter\s+name=[\"'](\w+)[\"']\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
FENCE_RE = re.compile(r"^```(?:json|python)?\s*|\s*```$", re.IGNORECASE)
UNQUOTED_KEY_RE = re.compile(r"([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")
SINGLE_QUOTED_STRING_RE = re.compile(r"'([^'\\]*)'")
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def skip_whitespace(text, start):
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def extract_balanced_object(text, start):
    """Return the substring of a balanced {...} object starting at start, or None."""
    start = skip_whitespace(text, start)
    if start < len(text) and text.startswith("```", start):
        newline = text.find("\n", start)
        if newline != -1:
            start = skip_whitespace(text, newline + 1)
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


def _normalize_args_text(args_str):
    text = (args_str or "").strip()
    text = FENCE_RE.sub("", text).strip()
    closer = text.rfind("</ARGS>")
    if closer != -1:
        text = text[:closer].strip()
    return text


def parse_tool_args(args_str):
    """Parse tool ARGS: JSON, single quotes, unquoted keys, or a Python dict literal."""
    text = _normalize_args_text(args_str)
    if not text:
        raise ValueError("Empty tool arguments")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = SINGLE_QUOTED_STRING_RE.sub(r'"\1"', text)
    repaired = UNQUOTED_KEY_RE.sub(r'\1"\2":', repaired)
    repaired = repaired.replace("True", "true").replace("False", "false").replace("None", "null")
    repaired = TRAILING_COMMA_RE.sub(r"\1", repaired)
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Tool arguments must be an object, got {type(parsed).__name__}")
    return parsed


def _parse_parameter_value(raw):
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def extract_invoke_calls(response):
    calls = []
    for match in INVOKE_RE.finditer(response):
        args = {}
        for param in PARAM_RE.finditer(match.group(2)):
            args[param.group(1)] = _parse_parameter_value(param.group(2))
        if args:
            calls.append({"name": match.group(1), "args": args})
    return calls


def extract_tool_calls(response):
    if not response:
        return []

    tool_calls = []
    seen = set()
    for match in FUNCTION_HEADER_RE.finditer(response):
        tool_name = match.group(1)
        args_str = extract_balanced_object(response, match.end())
        if args_str is None:
            print(f"Parameter parsing failed: no JSON object after <ARGS> for {tool_name}")
            continue
        try:
            args = parse_tool_args(args_str)
        except Exception as exc:
            preview = args_str[:120].replace("\n", " ")
            print(f"Parameter parsing failed for {tool_name}: {exc} | args={preview!r}")
            continue
        key = (tool_name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        tool_calls.append({"name": tool_name, "args": args})

    for call in extract_invoke_calls(response):
        key = (call["name"], json.dumps(call["args"], sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        tool_calls.append(call)
    return tool_calls


def has_config_parameter(func):
    return "config" in inspect.signature(func).parameters
