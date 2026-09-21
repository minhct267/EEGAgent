import os
import json
import time
import yaml
from openai import OpenAI
from tools import function_register
from tools.registerData import registerData
from tools.dataLoad import dataLoad, load_MDD_edf, load_Sleep_edf
from tools.baseInfo import baseInfo, get_age_factor
from planner_adapt import (
    CONTINUE_AFTER_TOOLS,
    DISCHARGE_TOOL_NAMES,
    continue_after_tools,
    filter_tool_schemas,
    preserve_planner_reasoning,
    tool_results_as_user,
)
from prompt import getSystemPrompt
from utils.parseCalling import extract_tool_call_failures, extract_tool_calls, has_config_parameter
from utils.messageMerge import messageMerge
from RAG.embedder import BGEEmbedder
from RAG.searcher import FaissSearcher
from llm_settings import get_planner_settings, planner_client, planner_extra_body, strip_think_tags


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
        self.api_key = api_key if api_key else settings.api_key
        self.base_url = base_url if base_url else settings.base_url
        self.model = model if model else settings.model
        self.timeout = settings.timeout
        self.reasoning_effort = settings.reasoning_effort
        self.extra_body = planner_extra_body(settings)
        self.preserve_reasoning = preserve_planner_reasoning(self.model, self.reasoning_effort)
        self.tool_results_as_user = tool_results_as_user(self.model)
        self._appended_assistant = False
        if api_key or base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            self.client = planner_client()

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
        )
        self.messages = [{'role': 'system', 'content': self.system_prompt}]

    def prepare_user_message(self, user_query: str):
        self.messages.append({'role': 'user', 'content': user_query})

    def call_model(self, model: str | None = None):
        used_model = model or self.model
        completion = self.client.chat.completions.create(
            model=used_model,
            messages=self.messages,
            timeout=self.timeout,
            extra_body=self.extra_body,
        )
        message = completion.choices[0].message
        raw_content = message.content or ""
        visible = strip_think_tags(raw_content)
        stored = raw_content if self.preserve_reasoning else visible
        self._appended_assistant = bool(stored.strip())
        if self._appended_assistant:
            self.messages.append({
                "role": "assistant",
                "content": stored
            })
        return visible

    def handle_tool_calls(self, response):
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
        messageMerge(function_return, self.messages, as_user=self.tool_results_as_user)
        if continue_after_tools(self.model) and not self.tool_results_as_user:
            self.messages.append({"role": "user", "content": CONTINUE_AFTER_TOOLS})
        return True

    def run(self, user_query, max_rounds=8):
        embedder = BGEEmbedder()
        searcher = FaissSearcher("RAG/faiss.index", "RAG/chunks.pkl")
        query_vector = embedder.encode([user_query])[0]
        threshold = self.config.get("similarity_threshold")
        results = searcher.search(query_vector, top_k=3)
        for rank, (text, score) in enumerate(results, 1):
            if score >= threshold:
                self.messages[0]['content'] += f"{rank}. {text}\n"

        self.prepare_user_message(user_query)

        total_rounds = 0
        total_time = 0.0
        local_tool_time = 0.0
        response = ""

        start_total = time.time()
        stopped_by_max_rounds = False
        empty_retries = 0
        while total_rounds < max_rounds:
            round_start = time.time()
            response = self.call_model()
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
            if empty_retries >= 1:
                break
            empty_retries += 1
            self.messages.append({
                "role": "user",
                "content": (
                    "The previous model turn was empty. Using the tool results already "
                    "in this conversation, write the final answer now. If you still need "
                    "a tool, emit <FUNCTION>/<ARGS> with valid JSON."
                ),
            })
        else:
            stopped_by_max_rounds = True

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
            "stopped_by_max_rounds": stopped_by_max_rounds
        }


if __name__ == "__main__":
    # Planner credentials and model come from .env (local Ollama by default).
    agent = EEGAgent(
        config_path="config/config.json",
        file_name="gped_049_a_6.edf",
        tool_names=DISCHARGE_TOOL_NAMES,
    )
    user_question = "Can epileptic discharges be observed within the first minute? If so, where?"
    print("Human:", user_question)
    result = agent.run(user_question)
    print("Assistant:", result["response"])
    print(
        f"rounds={result['rounds']} model_time={result['model_time']:.2f}s "
        f"local_tool_time={result['local_tool_time']:.2f}s total_time={result['total_time']:.2f}s"
    )
