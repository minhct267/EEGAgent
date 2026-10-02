import os
import json
import time
import yaml
from tools import function_register
from tools.registerData import registerData
from tools.dataLoad import dataLoad, load_MDD_edf, load_Sleep_edf
from tools.baseInfo import baseInfo, get_age_factor
from dataclasses import replace

from planner_adapt import (
    EMPTY_RETRY_MESSAGE,
    FORCE_FINAL_MESSAGE,
    HARNESS_AUTHORS_V2,
    filter_tool_schemas,
    preserve_planner_reasoning,
    truncate_at_stop,
)
from prompt import getSystemPrompt
from utils.parseCalling import (
    extract_tool_call_failures,
    extract_tool_calls,
    extract_tool_calls_authors,
    has_config_parameter,
)
from utils.messageMerge import messageMerge
from RAG.embedder import BGEEmbedder
from RAG.searcher import FaissSearcher
from llm_settings import (
    ContextOverflowError,
    default_planner_api,
    get_planner_settings,
    make_planner_client,
    planner_extra_body,
    strip_think_tags,
)


def resolve_eeg_path(data_path: str, file_name: str) -> str:
    """Prefer an existing path; otherwise join with config dataPath."""
    if os.path.exists(file_name):
        return file_name
    return os.path.join(data_path or "", file_name)


class EEGAgent:
    def __init__(
        self,
        config_path: str,
        file_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        tool_names: tuple[str, ...] | list[str] | None = None,
        harness=None,
        prompt_mode: str = "authors",
        rag_enabled: bool = True,
        rag_top_k: int = 3,
        sampling: dict | None = None,
    ):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        with open(os.path.join(self.config['report_template_path'], 'report_template.yaml'), 'r') as file:
            report_template = yaml.safe_load(file)
        self.file_path = resolve_eeg_path(self.config.get('dataPath', ''), file_name)

        # Load EEG with the TUEV bipolar montage by default.
        # data = load_MDD_edf(self.file_path, self.config)
        data = dataLoad(self.file_path, self.config)
        # data = load_Sleep_edf(self.file_path, self.config)
        registerData(data)

        self.info = baseInfo(self.file_path)
        self.tool_names = tool_names
        self.tool_schemas = filter_tool_schemas(function_register.export_tool_schemas(), tool_names)

        settings = get_planner_settings()
        if sampling:
            settings = replace(settings, **sampling)
        if api_key:
            settings = replace(settings, api_key=api_key)
        if base_url and base_url != settings.base_url:
            settings = replace(settings, base_url=base_url, api=default_planner_api(base_url))
        if model:
            settings = replace(settings, model=model)
        self.settings = settings
        self.api_key = settings.api_key
        self.base_url = settings.base_url
        self.model = settings.model
        self.timeout = settings.timeout
        self.reasoning_effort = settings.reasoning_effort
        self.num_ctx = settings.num_ctx
        self.extra_body = planner_extra_body(settings)
        self.harness = harness or HARNESS_AUTHORS_V2
        self.prompt_mode = prompt_mode
        self.rag_enabled = rag_enabled
        self.rag_top_k = rag_top_k
        self.preserve_reasoning = preserve_planner_reasoning(self.model, self.reasoning_effort)
        self._appended_assistant = False
        self.call_records = []
        self.harness_events = []
        self.retrieval = []
        self.client = make_planner_client(settings)

        prior_knowlwdge = self.config['prior knowledge'].copy()
        if 'age' in self.info.keys():
            age_factor = get_age_factor(self.info['age'], self.config['prior knowledge']['Age factor'])
            prior_knowlwdge['Age factor'] = age_factor
        self.system_prompt = getSystemPrompt(
            self.info,
            self.tool_schemas,
            prior_knowlwdge,
            report_template,
            planner_model=self.model,
            prompt_mode=self.prompt_mode,
        )
        self.messages = [{'role': 'system', 'content': self.system_prompt}]

    def prepare_user_message(self, user_query: str):
        self.messages.append({'role': 'user', 'content': user_query})

    def _usage_value(self, usage, name: str):
        if usage is None:
            return None
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    def call_model(self, model: str | None = None):
        used_model = model or self.model
        kwargs = {
            "model": used_model,
            "messages": self.messages,
            "timeout": self.timeout,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "seed": self.settings.seed,
            "extra_body": self.extra_body,
        }
        if self.harness.stop:
            kwargs["stop"] = list(self.harness.stop)
        api = getattr(self.settings, "api", "openai")
        started = time.time()
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except ContextOverflowError as exc:
            self.call_records.append({
                "model": used_model,
                "api": api,
                "latency_seconds": time.time() - started,
                "prompt_tokens": exc.prompt_tokens,
                "completion_tokens": 0,
                "finish_reason": "context_overflow",
                "stop": list(self.harness.stop),
                "stop_hit": False,
                "context_warning": True,
                "context_truncated": False,
                "num_ctx": exc.num_ctx or self.num_ctx,
            })
            self.harness_events.append("context_overflow")
            raise
        latency = time.time() - started
        choice = completion.choices[0]
        message = choice.message
        raw_content = message.content or ""
        visible = strip_think_tags(raw_content)
        visible, stop_hit = truncate_at_stop(visible, self.harness.stop)
        if self.preserve_reasoning:
            stored, stored_hit = truncate_at_stop(raw_content, self.harness.stop)
            stop_hit = stop_hit or stored_hit
        else:
            stored = visible
        usage = getattr(completion, "usage", None)
        prompt_tokens = self._usage_value(usage, "prompt_tokens")
        completion_tokens = self._usage_value(usage, "completion_tokens")
        context_warning = bool(
            prompt_tokens is not None and self.num_ctx and prompt_tokens >= 0.95 * self.num_ctx
        )
        # History only grows within a window, so a smaller prompt means the server dropped messages.
        previous = [item["prompt_tokens"] for item in self.call_records if item.get("prompt_tokens") is not None]
        context_truncated = bool(prompt_tokens is not None and previous and prompt_tokens < max(previous))
        self.call_records.append({
            "model": used_model,
            "api": api,
            "latency_seconds": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": getattr(choice, "finish_reason", None),
            "stop": list(self.harness.stop),
            "stop_hit": stop_hit,
            "context_warning": context_warning,
            "context_truncated": context_truncated,
            "num_ctx": self.num_ctx,
        })
        if stop_hit:
            self.harness_events.append("stop_hit")
        if context_warning:
            self.harness_events.append("context_warning")
        if context_truncated:
            self.harness_events.append("context_truncated")
        self._appended_assistant = bool(stored.strip())
        if self._appended_assistant:
            self.messages.append({
                "role": "assistant",
                "content": stored
            })
        return visible

    def _handle_tool_calls_authors(self, response):
        """The authors' loop: unknown tools and unparsable ARGS are skipped without feedback."""
        calls = extract_tool_calls_authors(response)
        if not calls:
            return False
        function_return = []
        for call in calls:
            function_name = call['name'].strip()
            args = call['args'].copy()
            function = function_register.get_function(function_name)
            if not function:
                print(f"No function named {function_name}")
                continue
            if has_config_parameter(function) and 'config' not in args:
                args['config'] = self.config
            try:
                output = function(**args)
                new_call = call.copy()
                new_call['return'] = output
                function_return.append(new_call)
            except Exception as e:
                print(f"{function_name} execution unsuccessful: {e}")
                new_call = call.copy()
                new_call['return'] = {
                    "error": repr(e),
                    "instruction": "The tool call failed. Revise the arguments to satisfy the tool constraints before calling again."
                }
                function_return.append(new_call)
        messageMerge(
            function_return,
            self.messages,
            as_user=self.harness.tool_result_role == "user",
            notes=self.prompt_mode == "strict",
        )
        return True

    def handle_tool_calls(self, response):
        if self.harness.tool_calls == "authors":
            return self._handle_tool_calls_authors(response)
        source = response
        if (
            getattr(self, "_appended_assistant", False)
            and self.messages
            and self.messages[-1].get("role") == "assistant"
        ):
            source = self.messages[-1].get("content") or response
        calls = extract_tool_calls(source)
        failures = extract_tool_call_failures(source)
        if not calls and not failures and source != response:
            calls = extract_tool_calls(response)
            failures = extract_tool_call_failures(response)
        if not calls and not failures:
            return False

        failed_instruction = (
            "The tool call failed. Revise the arguments to satisfy the tool constraints before calling again."
        )
        function_return = []
        for failure in failures:
            function_return.append({
                "name": failure["name"],
                "args": failure.get("args") or {},
                "return": {
                    "error": failure["error"],
                    "instruction": failed_instruction,
                },
            })
        for call in calls:
            function_name = call['name'].strip()
            args = call['args'].copy()
            function = function_register.get_function(function_name)
            if not function:
                print(f"No function named {function_name}")
                function_return.append({
                    "name": function_name,
                    "args": args,
                    "return": {
                        "error": f"No function named {function_name}",
                        "instruction": failed_instruction,
                    },
                })
                continue

            # Inject config when the tool signature expects it.
            if has_config_parameter(function) and 'config' not in args:
                args['config'] = self.config

            try:
                output = function(**args)
                new_call = call.copy()
                new_call['return'] = output
                function_return.append(new_call)
            except Exception as e:
                print(f"{function_name} execution unsuccessful: {e}")
                new_call = call.copy()
                new_call['return'] = {
                    "error": repr(e),
                    "instruction": "The tool call failed. Revise the arguments to satisfy the tool constraints before calling again."
                }
                function_return.append(new_call)
        if not function_return:
            return False
        messageMerge(
            function_return,
            self.messages,
            as_user=self.harness.tool_result_role == "user",
            notes=self.prompt_mode == "strict",
        )
        return True

    def _retrieve(self, user_query: str):
        self.retrieval = []
        if not self.rag_enabled:
            self.harness_events.append("rag_off")
            return
        embedder = BGEEmbedder()
        searcher = FaissSearcher("RAG/faiss.index", "RAG/chunks.pkl")
        query_vector = embedder.encode([user_query])[0]
        threshold = self.config.get("similarity_threshold")
        results = searcher.search(query_vector, top_k=self.rag_top_k)
        for rank, (text, score) in enumerate(results, 1):
            kept = threshold is None or score >= threshold
            self.retrieval.append({
                "rank": rank,
                "score": score,
                "kept": kept,
                "text": text,
            })
            if kept:
                self.messages[0]["content"] += f"{rank}. {text}\n"

    def run(self, user_query, max_rounds=8):
        self.call_records = []
        self.harness_events = []
        self._retrieve(user_query)
        self.prepare_user_message(user_query)

        total_rounds = 0
        total_time = 0.0
        local_tool_time = 0.0
        response = ""

        start_total = time.time()
        stopped_by_max_rounds = False
        empty_retries = 0
        forced_final = False
        context_overflow = False
        while total_rounds < max_rounds:
            round_start = time.time()
            try:
                response = self.call_model()
            except ContextOverflowError:
                # A longer history cannot fit either, so the window ends here without a final turn.
                context_overflow = True
                response = ""
                total_rounds += 1
                break
            round_end = time.time()
            total_rounds += 1
            total_time += (round_end - round_start)

            tool_start = time.time()
            has_more_tools = self.handle_tool_calls(response)
            local_tool_time += (time.time() - tool_start)

            if has_more_tools:
                continue
            if response.strip():
                break
            if not self.harness.empty_retry or empty_retries >= 1:
                break
            empty_retries += 1
            self.harness_events.append("empty_retry")
            self.messages.append({"role": "user", "content": EMPTY_RETRY_MESSAGE})
        else:
            stopped_by_max_rounds = True

        if stopped_by_max_rounds and self.harness.force_final:
            forced_final = True
            self.harness_events.append("force_final")
            self.messages.append({"role": "user", "content": FORCE_FINAL_MESSAGE})
            round_start = time.time()
            try:
                response = self.call_model()
            except ContextOverflowError:
                context_overflow = True
                response = ""
            total_rounds += 1
            total_time += time.time() - round_start

        total_duration = time.time() - start_total

        last_assistant = ""
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                last_assistant = message.get("content") or ""
                break

        return {
            "response": response,
            "raw_assistant": last_assistant,
            "rounds": total_rounds,
            "model_time": total_time,
            "local_tool_time": local_tool_time,
            "total_time": total_duration,
            "stopped_by_max_rounds": stopped_by_max_rounds,
            "forced_final": forced_final,
            "context_overflow": context_overflow,
            "harness": self.harness.name,
            "harness_events": list(self.harness_events),
            "calls": list(self.call_records),
            "retrieval": list(self.retrieval),
            "prompt_mode": self.prompt_mode,
            "rag_enabled": self.rag_enabled,
        }


if __name__ == "__main__":
    # Planner credentials and model come from .env (local Ollama by default).
    agent = EEGAgent(
        config_path="config/config.json",
        file_name="gped_049_a_6.edf",
    )
    user_question = "Can epileptic discharges be observed within the first minute? If so, where?"
    print("Human:", user_question)
    result = agent.run(user_question)
    print("Assistant:", result["response"])
    print(
        f"rounds={result['rounds']} model_time={result['model_time']:.2f}s "
        f"local_tool_time={result['local_tool_time']:.2f}s total_time={result['total_time']:.2f}s"
    )
