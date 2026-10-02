# System

EEGAgent is a single-LLM ReAct loop plus a local EEG toolbox. The paper is Zhao et al., arXiv:2511.09947v2 (27 Nov 2025). This tree is that system pointed at local Ollama.

## Loop

1. `EEGAgent` loads one EDF, preprocesses it, and stores the array in a process-global buffer. Tools never receive the waveform as an argument. They slice it by time.
2. A system prompt is built from the EDF header, age-band priors, channel table, and JSON schemas of the allowed tools (`prompt.py`).
3. The user question is embedded once. Up to three chunks with cosine similarity at least 0.6 are appended to the system prompt (`main.py`, `config.json`). Later rounds do not retrieve again.
4. The planner replies in text. `utils/parseCalling.py` extracts `<FUNCTION>` / `<ARGS>` (and MiniMax `<invoke>` blocks). Python runs the functions. `utils/messageMerge.py` writes the results back into the transcript.
5. The loop stops at eight rounds, or earlier when a non-empty reply contains no tool call. `authors_v2` then takes one more tool-free turn if the cap was hit. `authors_v1` does not. The returned text can still be a tool call when `stopped_by_max_rounds` is true and that extra turn was not used.

The chat API is not given a native `tools` argument. Tool use is the XML dialect in the assistant string.

Active loader in `main.py` is always the TUEV bipolar path: bandpass 0.5–70 Hz, notch 60 Hz, resample to 256 Hz, up to 22 TCP pairs (`tools/dataLoad.py`, `tools/polar.py`). `load_MDD_edf` and `load_Sleep_edf` exist and are commented out.

## What the paper specifies

The planner named in the paper is Qwen3-235B. The paper's retrieval sentence names Qwen3-Embedding-8B. The released embedding code is BGE-M3; see the table below. The toolbox in Table 1 is:

| Tool | Role |
| --- | --- |
| `normalAbnormal` | Whole record, normal vs abnormal |
| `eyemMuscle` | 1 s, one channel, eye vs muscle |
| `seizArtiBckg` | 1 s, one channel, seizure vs artifact vs background |
| `seizNormal` | 1 s, one channel, seizure vs non-seizure |
| `slowSeizBckg` | 10 s, all channels, slow vs seizure vs background |
| `baseInfo` | Header: id, sex, age, montage, duration |
| `compute_amplitude`, `compute_psd`, `compute_symmetry` | Windows of at most 60 s |

Detection is coarse-to-fine: a 10 s screen, then 1 s tools on the suspicious interval and channels. Reporting follows an ACNS-style template (Algorithm 1): 10 s segments, optional 1 s refinement, then a structured report.

The only quantitative result in the paper is TUEV event detection. TUAB perception, exploration, and report generation are case studies. TUSL is described as a dataset and is not given a number. Sleep staging and MDD are not in the paper.

## What this tree changes

| Piece | Paper | This tree |
| --- | --- | --- |
| Planner | Qwen3-235B API | `qwen3.8:27b` on local Ollama. See [models.md](models.md). |
| Embeddings | The paper text says Qwen3-Embedding-8B. The released code embeds with local BGE-M3 (CLS pooling, 1,024 dimensions, 653 chunks). | Ollama `bge-m3:latest` against that same index. The folder `RAG/sentenceModel/bge-m3` is not loaded at runtime. |
| Thinking | The authors' `call_model` is non-streaming, and its `enable_thinking` line is commented out. DashScope rejects that call unless thinking is off, so the released 235B run had thinking off. | The baseline also runs with thinking off: `OLLAMA_REASONING_EFFORT=none` and `think: false`. Context is `OLLAMA_NUM_CTX=65536`, sent through Ollama's native `/api/chat` with `truncate: false`. The `/v1` endpoint would ignore it. |
| Tool code and text | At `acd2e6a` the three 1 s tools scored the first montage rows and labeled them with the requested channels. | The tools score the requested channel rows. The authors prompt still shows the `acd2e6a` tool text, from `config/protocols/authors_tool_schemas.json`. The rewritten tool text is used only by `--prompt strict`. See [baseline.md](baseline.md). |
| `baseInfo` | A tool the agent calls | Run at load time and pasted into the prompt. Not an LLM tool. |
| Report template | Algorithm 1 fills `RAG/report_template.yaml` | The YAML is loaded. `getSystemPrompt` does not insert it. |
| Extra tools | Not in Table 1 | `reflectData`, `sleepStageModel`, `healthMDDModel` are registered. The baseline exposes every registered tool, which is what `acd2e6a` did. `--prompt strict` limits the prompt to the nine names in `DISCHARGE_TOOL_NAMES`. |
| TUEV question | The released eval script asks for seizures in a rounded window and for `(channel, start, end)` lines, with no 0.50 rule. | `tuev_authors_v2` uses that question. `--prompt strict` is the later local question: inspect every second, report a channel only when a 1 s tool has `seiz` as the top class and `seiz >= 0.50`, or write exactly `No events found`. |
| Eval windows | Not described | `TUEV_eval.py` asks only about intervals merged from `.rec` rows, including eye movement, artifact, and background. The model is told the time span. It is not told the channel. |

The `<FUNCTION>` / `<ARGS>` / `<RETURN>` / `<RESULT>` block is the authors' format from `acd2e6a`, not a local invention. Under `authors_v2` the same block is written as a user turn and generation stops at `<RETURN>`. Those switches are protocol flags, not model-name checks. The old Qwen3.8 continue-after-tools patch is gone. See [xml.md](xml.md).

## TUEV numbers

The local eval and how it differs from the paper's 69.30% / 44.77% are in [tuev.md](tuev.md).
