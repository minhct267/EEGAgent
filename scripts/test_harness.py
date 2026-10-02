"""Harness checks: stop truncation, tool-result placement, Ollama native options, context guards."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import llm_settings
from llm_settings import ContextOverflowError, OllamaNativeClient, PlannerSettings, default_planner_api
from main import EEGAgent
from planner_adapt import HARNESS_AUTHORS_V1, HARNESS_AUTHORS_V2, truncate_at_stop
from prompt import authors_tool_schemas, getSystemPrompt
from tools import function_register
from utils.messageMerge import messageMerge


def expect(condition: bool, detail: str) -> None:
    if not condition:
        raise SystemExit(f"[FAIL] {detail}")
    print(f"[PASS] {detail}")


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Usage:
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = 40


class _Completion:
    def __init__(self, content, prompt_tokens):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens)


class _Completions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.kwargs = kwargs
        if self.owner.raise_overflow:
            raise ContextOverflowError("too long", 70000, 65536)
        tokens = self.owner.prompt_tokens.pop(0) if self.owner.prompt_tokens else 128
        return _Completion(self.owner.scripted, tokens)


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class FakeClient:
    def __init__(self, scripted, prompt_tokens=None, raise_overflow=False):
        self.scripted = scripted
        self.prompt_tokens = list(prompt_tokens or [])
        self.raise_overflow = raise_overflow
        self.kwargs = None
        self.chat = _Chat(self)


TOOL_CALL = (
    "<FUNCTION> seizureNormalModel_OneSecond\n"
    "<ARGS> {\"name\": [\"FP1-F7\"], \"start\": 0, \"end\": 1}\n"
    "<RETURN>\n[{'fake': 1}]\n<RESULT>"
)


def _agent(harness, client=None, prompt_mode="authors"):
    agent = EEGAgent.__new__(EEGAgent)
    agent.model = "fake"
    agent.timeout = 5
    agent.settings = PlannerSettings(
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="fake",
        timeout=5,
        reasoning_effort="none",
        num_ctx=65536,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        seed=0,
        api="ollama_native",
    )
    agent.extra_body = {"options": {"num_ctx": 65536}}
    agent.harness = harness
    agent.prompt_mode = prompt_mode
    agent.preserve_reasoning = False
    agent.num_ctx = 65536
    agent.rag_enabled = False
    agent.config = {}
    agent.messages = [{"role": "system", "content": "system"}]
    agent.call_records = []
    agent.harness_events = []
    agent.retrieval = []
    agent.client = client or FakeClient(TOOL_CALL)
    return agent


def check_stop_and_placement() -> None:
    trimmed, hit = truncate_at_stop("call <RETURN> fake", ("<RETURN>",))
    expect(hit and trimmed == "call", "Stop marker is removed from the stored text")

    agent = _agent(HARNESS_AUTHORS_V2)
    agent.messages.append({"role": "user", "content": "question"})
    visible = agent.call_model()
    expect(agent.client.kwargs["stop"] == ["<RETURN>"], "authors_v2 sends stop=['<RETURN>']")
    expect(agent.call_records[-1]["stop_hit"] is True, "Stop sequence hit is recorded")
    expect("<RETURN>" not in visible and "fake" not in visible, "Simulated return is not returned to the loop")
    expect("<RETURN>" not in agent.messages[-1]["content"], "Assistant history has no simulated <RETURN>")
    expect(agent.client.kwargs["temperature"] == 0.7 and agent.client.kwargs["seed"] == 0, "Sampling arguments are sent")

    v1 = _agent(HARNESS_AUTHORS_V1)
    v1.messages.append({"role": "user", "content": "question"})
    v1.call_model()
    expect("stop" not in v1.client.kwargs, "authors_v1 does not send a stop sequence")
    expect("<RETURN>" in v1.messages[-1]["content"], "authors_v1 keeps the original completion text")

    messages = [{"role": "assistant", "content": "<FUNCTION>\nseizureNormalModel_OneSecond"}]
    messageMerge(
        [{"name": "seizureNormalModel_OneSecond", "args": {"start": 0}, "return": {"ok": True}}],
        messages,
        as_user=True,
    )
    expect(messages[-1]["role"] == "user", "Tool results are a new user turn")
    expect(messages[0]["content"] == "<FUNCTION>\nseizureNormalModel_OneSecond", "The assistant call is not spliced")
    expect("<RETURN>" in messages[-1]["content"] and "<RESULT>" in messages[-1]["content"], "User turn uses the result block")


def check_notes_and_parsers() -> None:
    item = [{"name": "seizureNormalModel_OneSecond", "args": {"start": 0}, "return": {"ok": True}}]
    plain, noted = [], []
    messageMerge(item, plain, as_user=True, notes=False)
    messageMerge(item, noted, as_user=True, notes=True)
    expect("<NOTE>" not in plain[-1]["content"], "No <NOTE> block when notes are off (authors prompt)")
    expect("<NOTE>" in noted[-1]["content"], "<NOTE> block when notes are on (strict prompt)")

    spliced = [{"role": "assistant", "content": "call"}]
    messageMerge(item + item, spliced, notes=False)
    expect(spliced[-1]["content"].startswith("call\n\n<FUNCTION>") and "<RESULT><FUNCTION>" in spliced[-1]["content"],
           "authors_v1 splice concatenates blocks like the authors' messageMerge")

    single_quoted = "<FUNCTION> fakeTool\n<ARGS> {'start': 0}\n"
    v2 = _agent(HARNESS_AUTHORS_V2)
    v2.messages.append({"role": "assistant", "content": single_quoted})
    v2._appended_assistant = True
    expect(v2.handle_tool_calls(single_quoted), "authors_v2 repairs single-quoted ARGS and returns an error turn")
    expect("No function named fakeTool" in v2.messages[-1]["content"], "authors_v2 reports an unknown tool")

    v1 = _agent(HARNESS_AUTHORS_V1)
    v1.messages.append({"role": "assistant", "content": single_quoted})
    expect(not v1.handle_tool_calls(single_quoted), "authors_v1 drops single-quoted ARGS like the authors' parser")
    unknown = "<FUNCTION> fakeTool\n<ARGS> {\"start\": 0}\n"
    v1.messages.append({"role": "assistant", "content": unknown})
    expect(v1.handle_tool_calls(unknown), "authors_v1 continues after an unknown tool")
    expect("No function named" not in v1.messages[-1]["content"], "authors_v1 gives no feedback for an unknown tool")


def check_authors_prompt() -> None:
    live = function_register.export_tool_schemas()
    frozen = authors_tool_schemas(live)
    expect([s["function"]["name"] for s in frozen] == [s["function"]["name"] for s in live],
           "Authors tool schemas keep the live order")
    info = {"data_duration": 30}
    authors = getSystemPrompt(info, live, {}, None, prompt_mode="authors")
    strict = getSystemPrompt(info, live, {}, None, prompt_mode="strict")
    expect("Use this tool to classify the type of artifact" in authors, "Authors prompt shows the authors' tool text")
    expect("Binary artifact-TYPE classifier" not in authors, "Authors prompt omits the rewritten tool text")
    expect("Binary artifact-TYPE classifier" in strict, "Strict prompt keeps the rewritten tool text")


def check_native_client() -> None:
    expect(default_planner_api("http://127.0.0.1:11434/v1") == "ollama_native", "Local Ollama uses the native API")
    expect(default_planner_api("https://dashscope.aliyuncs.com/compatible-mode/v1") == "openai", "Other hosts use /v1")

    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, payload=json)
        if json["messages"][-1]["content"] == "overflow":
            body = ('{"error":"{\\"error\\":{\\"code\\":400,\\"message\\":\\"request (70000 tokens) exceeds the '
                    'available context size (65536 tokens)\\",\\"type\\":\\"exceed_context_size_error\\",'
                    '\\"n_prompt_tokens\\":70000,\\"n_ctx\\":65536}}"}')
            return SimpleNamespace(status_code=400, text=body, json=lambda: {})
        data = {"message": {"content": "ok"}, "done_reason": "stop", "prompt_eval_count": 11, "eval_count": 2}
        return SimpleNamespace(status_code=200, text="", json=lambda: data)

    original = llm_settings.httpx.post
    llm_settings.httpx.post = fake_post
    try:
        client = OllamaNativeClient("http://127.0.0.1:11434/v1")
        completion = client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=0.8,
            seed=0,
            stop=["<RETURN>"],
            extra_body={"reasoning_effort": "none", "think": False,
                        "options": {"num_ctx": 65536, "top_k": 20}},
        )
        payload = sent["payload"]
        expect(sent["url"] == "http://127.0.0.1:11434/api/chat", "Native client posts to /api/chat")
        expect(payload["options"]["num_ctx"] == 65536 and payload["options"]["top_k"] == 20,
               "num_ctx and top_k are sent as native options")
        expect(payload["options"]["stop"] == ["<RETURN>"] and payload["options"]["seed"] == 0, "Stop and seed are options")
        expect(payload["truncate"] is False and payload["think"] is False, "truncate=false and think=false are sent")
        expect(completion.choices[0].message.content == "ok" and completion.usage.prompt_tokens == 11,
               "Native response maps to choices/usage")
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}],
                                       extra_body={"reasoning_effort": "medium"})
        expect(sent["payload"]["think"] == "medium", "reasoning_effort medium is sent as think='medium'")
        try:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "overflow"}])
        except ContextOverflowError as exc:
            expect(exc.prompt_tokens == 70000 and exc.num_ctx == 65536, "Overflow raises ContextOverflowError with sizes")
        else:
            expect(False, "Overflow raises ContextOverflowError with sizes")
    finally:
        llm_settings.httpx.post = original


def check_context_guards() -> None:
    agent = _agent(HARNESS_AUTHORS_V2, FakeClient("No events found", prompt_tokens=[4000, 9000, 120]))
    agent.messages.append({"role": "user", "content": "q"})
    for _ in range(3):
        agent.call_model()
    flags = [call["context_truncated"] for call in agent.call_records]
    expect(flags == [False, False, True], "A shrinking prompt count is flagged context_truncated")

    overflow = _agent(HARNESS_AUTHORS_V2, FakeClient("", raise_overflow=True))
    result = overflow.run("question", max_rounds=3)
    expect(result["context_overflow"] and "context_overflow" in result["harness_events"],
           "run() ends the window on context overflow")
    expect(result["calls"][-1]["finish_reason"] == "context_overflow", "Overflow call is logged")


def main() -> None:
    check_stop_and_placement()
    check_notes_and_parsers()
    check_authors_prompt()
    check_native_client()
    check_context_guards()
    print("All harness checks passed.")


if __name__ == "__main__":
    main()
