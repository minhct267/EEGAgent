# TUEV baseline

Protocol id: `tuev_authors_v2` in [config/protocols/tuev_authors_v2.json](../config/protocols/tuev_authors_v2.json).

This is the run to compare later planners against. It is not the paper's 69.30% / 44.77%, and it is not the already finished `runs/tuev_agent_ollama` folder. That folder used a different question, a nine-tool subset, model-specific loop patches, and a context that was not the one it asked for (see Context below). Its numbers are in the table below as the confounded local run.

## Frozen settings

| Piece | Setting |
| --- | --- |
| Question | Authors' text from git `acd2e6a`. Times are `round(start)` and `round(end)`. |
| Tools | Every registered tool. The prompt shows the schemas the authors' commit registered, frozen in [config/protocols/authors_tool_schemas.json](../config/protocols/authors_tool_schemas.json) by `scripts/export_authors_tool_schemas.py`. |
| System prompt | Authors' prompt from `acd2e6a`, including its original wording. RAG lines are appended the same way, unlabeled, top 3, cosine at least 0.6. |
| Tool results | Authors' `<FUNCTION>/<ARGS>/<RETURN>/<RESULT>` block with no `<NOTE>` lines. Notes are a `--prompt strict` feature. |
| Harness | `authors_v2` for every planner: stop at `<RETURN>`, tool-result block as a new user turn, one empty-reply retry, one tool-free turn if round 8 is reached. Bad ARGS and unknown tool names are returned to the model as errors. |
| Planner | `qwen3.8:27b` through Ollama's native `/api/chat` (`OLLAMA_PLANNER_API=ollama_native`). Thinking off (`think: false`). |
| Sampling | temperature 0.7, top_p 0.8, top_k 20, seed 0. These match Qwen's non-thinking defaults. |
| Context | `num_ctx` 65536, sent with `truncate: false`. A prompt that does not fit ends the window with status `context_overflow`. A call at or above 95% of the cap is flagged `context_warning`. A prompt count lower than an earlier call in the same window is flagged `context_truncated`. |
| Retrieval | Ollama `bge-m3:latest` and the existing 1,024-d FAISS index. Query, chunks, and scores are stored on the window. |
| Scorer | Coverage at 0.7, unchanged, plus IoU > 0.7, per-class recall, event precision and F1, zero-GT reports, file-level bootstrap 95% interval (1,000 draws, seed 0). |

Each run writes `manifest.json` with the git commit, protocol, prompt hash, planner API, planner and embedder digests, and the SHA-256 of `RAG/faiss.index`, `RAG/chunks.pkl`, and the frozen authors' tool schemas. At the end it writes `run_health.json`: answer status counts, harness events, errors, the largest prompt, the context Ollama actually loaded, and the windows that overflowed or were truncated.

### Why the native API

Ollama's OpenAI-compatible `/v1` endpoint (checked on Ollama 0.35.0) ignores `extra_body.options`. A `/v1` request with `num_ctx` 65536 left the model at the server default of 32,768 tokens, and `top_k` was not applied either. Above that size Ollama silently drops the oldest messages. A 48k-token test prompt came back as 46 prompt tokens, and the model no longer saw the first user turn. `/api/chat` does load the requested context (65,536 confirmed in `/api/ps`, 20.5 GB on the two L4s), and with `truncate: false` it returns HTTP 400 instead of dropping messages.

Ollama strips the stop string from both APIs, so `stop_hit` stays 0 on Ollama. The stop sequence still works; the smoke check confirms no assistant turn contains `<RETURN>`. `stop_hit` only fires on a server that echoes the stop text.

## Differences from the authors' 235B run

The protocol keeps the authors' question, prompt, tool text, and result blocks. These differences remain and should be stated next to any 235B comparison:

| Difference | Effect |
| --- | --- |
| Channel-index fix in the three 1-second tools | At `acd2e6a`, `eyemMuscleModel_OneSecond`, `seizureArtiBckgModel_OneSecond`, and `seizureNormalModel_OneSecond` scored `data[:, ...]`, the first rows of the montage, then labeled row `j` with the `j`-th requested channel. Unless the request was a prefix of the montage order (for example all 22 channels), a channel's probabilities came from a different channel. The current tools score `data[ids, ...]`, with duplicates removed. The 235B run therefore saw different tool outputs for subset requests. |
| Eye-vs-muscle key | The authors' key `Eyem movement` is now `Eye movement`. |
| Harness `authors_v2` | Stop sequence, user-turn results, empty retry, forced final turn, and error feedback for bad calls. The authors' run used none of these. `--harness authors_v1` restores the authors' splice, parser (non-greedy regex and `json.loads`), and silent skipping of bad calls. |
| Planner backend | DashScope `qwen3-235b-a22b` with `enable_thinking=false`, against local `qwen3.8:27b` Q4_K_M. |

## What E0 already measured

Re-scored with `scripts/rescore_tuev.py` on 2 Oct 2026. Coverage hits match the metrics stored in each `agent_predictions.jsonl` (`sanity: true`).

The authors' run is git `acd2e6a:runs/tuev_agent` (`qwen3-235b-a22b`, 36 files, 120 windows). The local run is `runs/tuev_agent_ollama` (`qwen3.8:27b`). The 36-file rows use the same files and the same ground truth (447 positive events).

| Run | Files | Coverage hits | IoU > 0.7 hits | Unmatched reports | Event F1 |
| --- | --- | --- | --- | --- | --- |
| Authors' 235B, `acd2e6a` | 36 | 330/447 = 73.83% (95% CI 60.4–83.9) | 146/447 = 32.66% | 506/814 = 62.16% | 0.500 |
| Local qwen3.8, same 36 files | 36 | 260/447 = 58.17% (95% CI 45.5–68.9) | 122/447 = 27.29% | 423/694 = 60.95% | 0.467 |
| Local qwen3.8, full eval split | 159 | 1,422/2,736 = 51.97% (95% CI 43.0–59.9) | 692/2,736 = 25.29% | 1,734/3,418 = 50.73% | 0.506 |

Per-class coverage recall on the paired 36 files:

| Class | Events | 235B | qwen3.8 |
| --- | --- | --- | --- |
| SPSW | 50 | 33 (66.0%) | 37 (74.0%) |
| GPED | 230 | 190 (82.6%) | 146 (63.5%) |
| PLED | 167 | 107 (64.1%) | 77 (46.1%) |

Full-split local recall, same coverage rule: SPSW 112/216, GPED 936/1,633, PLED 374/887. Ninety files have no positive event and still produce 308 reports.

Re-parsing the saved answers with channel canonicalization changes the full local coverage count from 1,422 to 1,424 (file `pled_023_a_` only). The 235B hit count stays 330. Each run has one answer span whose channel is not in the TCP map: one in the 235B run, and one in the local run (inside the 36-file subset). Those spans are logged as `invalid_channel` and are not scored. The table above uses the stored parses, which are the numbers the original logs claimed.

The paper reports 69.30% hit rate and 44.77% false rate under "IoU > 0.7". The released scorer is coverage, not IoU. The released 235B run is 36 files, not 159, and its coverage hit rate is 73.83%, not 69.30%. Those paper figures are not a reproducible baseline.

## Runs still to execute

Exact commands, resume behavior, time, and the numbers to check are in [baseline_run.md](baseline_run.md). The eval commands there pass `--sleep 0`; the default without that flag is 5 seconds per window.

| Id | Command | Role |
| --- | --- | --- |
| E1 | `python TUEV_oracle_run.py --rule both --threshold 0.5 --threshold 0.7 --out-dir runs/tuev_oracle` | Detector ceiling, no LLM. |
| E2 | `python TUEV_eval.py --protocol tuev_authors_v2 --seed 0 --out-dir runs/tuev_authors_v2` | Primary qwen3.8 baseline, 159 files. |
| E3 | Same command with `--seed 1` and `--seed 2`, and `--file-list config/protocols/authors_235b_files.txt` | Spread across seeds on the 36 files that have a 235B run. |
| E4 | `--harness authors_v1`, `--prompt strict`, `--rag off`, `--think medium`, each with that file list | One factor at a time. `--prompt strict` also switches to the 0.50 question, the nine discharge tools, the rewritten tool text, and `<NOTE>` lines. |

Headline comparisons after E2: qwen3.8 against the 235B row on the 36 files, and qwen3.8 on 159 files against the oracle ceiling.

## Smoke

`python scripts/baseline_smoke.py` passed on 2 Oct 2026, after the switch to `/api/chat`. Unit checks cover:

- stop-sequence truncation
- native options (`num_ctx`, `top_k`, `stop`, `seed`, `truncate: false`, `think: false`)
- overflow and truncation flags
- `<NOTE>` placement
- the authors' parser under `authors_v1`
- the authors' tool text in the prompt

The live check used `bckg_014_a_` and `bckg_024_a_` (one window each, no positive labels). Five planner calls went through `ollama_native` with `stop: ["<RETURN>"]`. No assistant turn contained `<RETURN>`, and no transcript contained `<NOTE>`. Both system prompts showed the authors' tool text. `/api/ps` reported a loaded context of 65,536. The largest prompt was 18,378 tokens, prompt counts grew on every call, and there was no overflow or truncation. Resume left both transcripts in place.

These two files are a harness check, not a score. One window answered with two tuples. The other answered in prose that no channel qualified, which is status `unparseable` and contributes no events: the authors' question gives no explicit "no events" wording. Seed 0 is repeatable through `/api/chat` (same output twice; seed 1 differs).

Output paths from E0:

- `runs/tuev_agent_ollama/metrics_v2.json`
- `runs/rescore/acd2e6a/metrics_v2.json`
- `runs/rescore/local_on_authors36/metrics_v2.json`
