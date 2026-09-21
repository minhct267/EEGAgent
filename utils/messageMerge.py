import json

from planner_adapt import TOOL_RESULT_NOTES


def format_tool_results(function_return):
    blocks = []
    for item in function_return:
        name = item["name"]
        note = TOOL_RESULT_NOTES.get(name)
        block = (
            f"<FUNCTION>\n{name}\n"
            f"<ARGS>\n{json.dumps(item['args'], ensure_ascii=False, indent=2)}\n"
            f"<RETURN>\n{json.dumps(item['return'], ensure_ascii=False, default=str, indent=2)}\n"
        )
        if note:
            block += f"<NOTE>\n{note}\n"
        block += "<RESULT>"
        blocks.append(block)
    return "\n\n".join(blocks)


def messageMerge(function_return, messages, as_user=False):
    tool_response_text = format_tool_results(function_return)
    if not tool_response_text:
        return
    if as_user:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Tool results are below. Treat them as evidence. "
                    "Do not ignore seizure or discharge scores because "
                    "eyemMuscleModel_OneSecond muscle probability is high.\n\n"
                    f"{tool_response_text}"
                ),
            }
        )
        return
    messages[-1]["content"] += "\n\n" + tool_response_text
