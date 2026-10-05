# Pipelines

Three TUEV setups are easy to mix up. Only the first one is what this tree runs by default. The other two are saved logs.

## Current: Qwen3.8 on Ollama

Protocol file: [config/protocols/tuev_authors_v2.json](../config/protocols/tuev_authors_v2.json).

| Piece | Value |
| --- | --- |
| Planner | `qwen3.8:27b`. Manifest: GGUF Q4_K_M, family `qwen35`, 27.3B parameters |
| API | `ollama_native`. The client posts to Ollama `/api/chat`. The configured base URL is still `http://127.0.0.1:11434/v1`; the native client strips `/v1` |
| Why not `/v1` | Ollama's OpenAI-compatible `/v1` ignores `options` such as `num_ctx` and `top_k`. `/api/chat` applies them. `truncate: false` makes an over-long prompt fail instead of dropping old messages |
| Context | `num_ctx` 65536. The finished baseline loaded 65536 (`run_health.json`) |
| Thinking | `reasoning_effort` `none`, sent as `think: false` |
| Sampling | temperature 0.7, top_p 0.8, top_k 20, seed 0 unless `--seed` overrides it |
| Question | Authors' text in [tuev_protocol.py](../tuev_protocol.py): seizures between rounded start and end, one `(channel, start, end)` line per event. No 0.50 rule and no "No events found" sentence |
| Tools | `tool_names` is null, so every registered tool is allowed. The prompt shows the schemas from git `acd2e6a`, stored in [authors_tool_schemas.json](../config/protocols/authors_tool_schemas.json) |
| Harness | `authors_v2` for every model name. Stop at `<RETURN>`. Tool results are a new user turn. One empty-reply retry. One tool-free turn if round 8 is reached. Bad ARGS and unknown tools are returned as errors |
| Retrieval | On. Ollama `bge-m3:latest` (manifest: F16, BERT, 566.70M) against the existing FAISS index. Top 3, cosine at least 0.6 |
| Output | `runs/tuev_authors_v2` for seed 0 on all 159 files |

A non-Ollama `OLLAMA_PLANNER_BASE_URL` selects the OpenAI client (`default_planner_api` in [llm_settings.py](../llm_settings.py)). That path does not send native Ollama `num_ctx`. DashScope is such a host. This tree does not call it unless that URL is set.

## Released DashScope run: Qwen3-235B

Source: git `acd2e6a:runs/tuev_agent`, rescored to `runs/rescore/acd2e6a/metrics_v2.json`. Model string in that comparison: `qwen3-235b-a22b`. 36 files, 120 windows, 447 positive events. The file list is [config/protocols/authors_235b_files.txt](../config/protocols/authors_235b_files.txt).

That commit is not what `TUEV_eval.py` runs now. On `acd2e6a` the three 1-second tools set `x = data[:, start:end]` and then attach `probs[j]` to the `j`-th requested channel name. The current tools set `x = data[ids, start:end]`, so the scored rows are the requested channels. The authors' output key `Eyem movement` is now `Eye movement`. Harness `authors_v2` (stop sequence, user-turn results, empty retry, forced final turn, error feedback) was not that commit's loop. `--harness authors_v1` puts the splice and the strict `json.loads` parser back.

The paper's retrieval sentence names Qwen3-Embedding-8B. The released embedding code and this tree both use BGE-M3.

Do not rerun 235B on this harness and call it a reproduction of `acd2e6a`. The saved log is the 235B number in [baseline.md](baseline.md).

## Earlier Qwen3.8:27B folder

`runs/tuev_agent_ollama` is also `qwen3.8:27b`, 159 files, 512 windows. All 512 stored questions are the current strict question (`seiz >= 0.50`, or exactly `No events found`). The folder has no `manifest.json`, so API, context length, tool list, and harness are not in the log. Its coverage is 1,422/2,736. That is not the `tuev_authors_v2` score.

The current tree has no model-name branch that appends a "continue after tools" user turn. Harness choice is the protocol flag.

## Switches on the current code

Each flag changes one thing from `tuev_authors_v2`. Every finished run in this table is Qwen3.8:27B on local Ollama. None of them calls the DashScope API. The 36-file runs use seed 0, except the two extra seeds. The primary baseline is the 159-file row.

| Flag | What changes | Finished directory |
| --- | --- | --- |
| none | Authors' question, all tools, `authors_v2`, RAG on, `think: false`, seed 0 | `runs/tuev_authors_v2` (159 files) |
| `--seed 1` or `--seed 2` | Sampling seed only | `runs/tuev_authors_v2_36_seed1`, `..._seed2` |
| `--harness authors_v1` | Results spliced onto the assistant message. No stop, no empty retry, no forced final turn. Bad calls are dropped with no feedback | `runs/tuev_ablate_harness_v1` |
| `--prompt strict` | Strict question, the nine names in `DISCHARGE_TOOL_NAMES`, live tool text, and `<NOTE>` lines after results. Sleep and MDD tools are omitted | `runs/tuev_ablate_strict` |
| `--rag off` | No retrieved chunks | `runs/tuev_ablate_rag_off` |
| `--think medium` | `reasoning_effort` `medium` | `runs/tuev_ablate_think_medium` |

`DISCHARGE_TOOL_NAMES` is the three feature tools, `reflectData`, and the five detectors other than sleep and MDD.

## Oracle

[TUEV_oracle_run.py](../TUEV_oracle_run.py) loads each EDF, calls `seizureNormalModel_OneSecond` or `seizureArtiBckgModel_OneSecond` on the same rounded windows, and keeps a channel-second when `seiz` is at least the threshold. No planner, so this is neither Qwen3.8:27B nor Qwen3-235B. Same scorer as the Qwen3.8:27B (local Ollama) eval. Outputs:

- `runs/tuev_oracle/seizNormal_t0.5`
- `runs/tuev_oracle/seizNormal_t0.7`
- `runs/tuev_oracle/seizArtiBckg_t0.5`
- `runs/tuev_oracle/seizArtiBckg_t0.7`

The authors' question, used by Qwen3.8:27B (local Ollama), does not give that planner this threshold. The oracle is the ceiling of a fixed rule, not of every way Qwen3.8:27B might read the tools. It was not run for Qwen3-235B (DashScope).

## Sleep and MDD

Those agent scripts are not a second planner pipeline. See [overview.md](overview.md).
