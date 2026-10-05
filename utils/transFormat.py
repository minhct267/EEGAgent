"""Compact JSON helpers for text that is embedded in a prompt."""
import json
def format_json_for_prompt(data):
    """Dump JSON on one line so it can sit inside a prompt."""
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))