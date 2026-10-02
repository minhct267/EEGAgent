import json

from planner_adapt import TOOL_RESULT_NOTES


def format_tool_results(function_return, notes=True, separator="\n\n"):
    blocks = []
    for item in function_return:
        name = item["name"]
        note = TOOL_RESULT_NOTES.get(name) if notes else None
        block = (
            f"<FUNCTION>\n{name}\n"
            f"<ARGS>\n{json.dumps(item['args'], ensure_ascii=False, indent=2)}\n"
            f"<RETURN>\n{json.dumps(item['return'], ensure_ascii=False, default=str, indent=2)}\n"
        )
        if note:
            block += f"<NOTE>\n{note}\n"
        block += "<RESULT>"
        blocks.append(block)
    return separator.join(blocks)


def messageMerge(function_return, messages, as_user=False, notes=True):
    if as_user:
        tool_response_text = format_tool_results(function_return, notes=notes)
        if tool_response_text:
            messages.append({"role": "user", "content": tool_response_text})
        return
    # The authors' splice: blocks are concatenated after one blank line, even when empty.
    messages[-1]["content"] += "\n\n" + format_tool_results(function_return, notes=notes, separator="")
