# XML dialect

The `<FUNCTION>` / `<ARGS>` / `<RETURN>` / `<RESULT>` transcript is the authors' design in git `acd2e6a` (`utils/messageMerge.py` appended the block to the assistant message, and the closer is `<RESULT>`, not `</RETURN>`). It is not a local patch. Qwen3-235B finished that loop in two rounds on 108 of 120 windows in the released run. The released run did not save transcripts, so it is not possible to check whether 235B also wrote a simulated `<RETURN>`.

Qwen3.8-27B can emit the call XML. In the saved local run it often did not continue after the result block was written back into its own assistant message. The counts below are from that run, which already included a continue-after-tools patch, and they cover only the 202 windows whose transcripts survived. See the resume bug in [tuev.md](tuev.md).

The baseline does not special-case the model name. Harness `authors_v2` stops generation at the string `<RETURN>` and writes the same result block as a new user turn. One empty reply is retried once. If round 8 is reached, one tool-free turn asks for the final tuples. `--harness authors_v1` restores the authors' splice, with no stop sequence and no extra turns. It also restores the authors' parser, so bad ARGS and unknown tool names are skipped without feedback. That ablation is what measures the dialect, and it has not been run yet.

## Where it is recorded

Commit `9580e09` (21 Sep 2026, "Use qwen3.8 local and align code") added `planner_adapt.py`. The saved run used a function selected only when the model name matched `qwen3.8`. That function is not in the current tree:

```python
def continue_after_tools(model):
    """Qwen3.8 often emits an empty turn after tool merge; ask it to continue."""
```

In that commit, `main.py` appended `CONTINUE_AFTER_TOOLS` as a user message after every successful tool merge for that model. The message told the model to keep emitting `<FUNCTION>` / `<ARGS>`, or to write tuples / `No events found`. MiniMax, in the same commit, received tool results as a user turn instead of the continue sentence. The current harness uses the user-turn placement for every planner and does not append the continue sentence.

## What the model is shown

The system prompt says the tool returns `<RETURN> tool output` (`prompt.py`).

In the saved local run, `utils/messageMerge.py` does not open a tool role. It appends this block to `messages[-1]`, which is the assistant turn that made the call. The `<NOTE>` line was a local addition; the authors' block has no note. The baseline's authors prompt omits it, and only `--prompt strict` adds it.

```text
<FUNCTION>
tool_name
<ARGS>
{json}
<RETURN>
{json}
<NOTE>
optional scorer note
<RESULT>
```

There is no `</RETURN>`. The closing marker is `<RESULT>`. The block sits in the assistant role, so the next completion sees the result as text the assistant already produced. With no new user turn, the next Qwen3.8 completion is often empty, the loop sees no tool call and no answer, and it stops. The continue user turn is what makes the model take another step.

## What the TUEV logs show

The saved run already includes the patch. Of 512 windows, 202 have transcripts and 310 have `messages: null`.

- All 202 transcripts contain at least one "Continue from the tool results..." user message.
- Those transcripts contain 573 assistant messages. 377 of them include a `<RETURN>` block, and 376 include the code's `<RESULT>` closer. The model is writing `<FUNCTION>` calls, and the runtime is splicing real results.
- No `raw_response` uses an OpenAI `tool_calls` payload.
- Across the assistant messages that immediately follow a CONTINUE turn (376 such turns, more than one per window when several tool rounds occur): 167 start with `<FUNCTION>`, 8 contain `<FUNCTION>` later, 121 are event tuples, 54 are `No events found`, 21 are other prose, and 5 are empty.

So the patch is doing the work the comment describes. It does not remove the failure. Ten windows still hit round 8 while `stopped_by_max_rounds` is true and contribute no parsed events. Five CONTINUE follow-ups are empty.

Example, `runs/tuev_agent_ollama/bckg_037_a_/messages/bckg_037_a_.messages.json`, candidate 1, window 41–49 s: the assistant message starts with `<FUNCTION> slowSeizBckgModel_TenSeconds`, the runtime splices the real scores at the end and closes with `<RESULT>`, the next role is the CONTINUE user line, and only the assistant turn after that line is `No events found`.

Forty-three of the 377 result messages also contain a compact single-quoted `<RETURN>` (a Python-style list) in addition to the indented JSON block the runtime writes. In the example above, the model wrote a simulated return in the same generation as the call, instead of stopping after `<ARGS>` and waiting for execution.

## Practical consequence

The saved 58.17% versus 73.83% comparison already includes the CONTINUE patch on the qwen3.8 side and the original splice on the 235B side. It does not isolate the dialect. The dialect cost is the `--harness authors_v1` ablation on the 36-file list, under the frozen sampler and the authors' question.
