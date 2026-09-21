# EEGAgent runbook

Commands assume the repo root and conda env `bci`. Copy `.env.example` to `.env` before the first run. Do not commit `.env`.

## Datasets

Keep these local paths. The code reads them; it does not write into them.

- TUEV: `D:\Datasets\TUH-EEG\TUEV`
- TUAB: `D:\Datasets\TUH-EEG\TUAB`
- TUEV official eval split used by `TUEV_eval.py`: `D:\Datasets\TUH-EEG\TUEV\v2.0.1\edf\eval`

A small sample EDF for smoke tests lives in the repo at `data/gped_049_a_6.edf`.

## Config

Two layers. Change `.env` to switch models and eval paths. Change `config/config.json` only for signal priors (sample rate, filters, montage text).

| File | What it controls |
| --- | --- |
| `.env` | Planner URL/model/key, embedding URL/model, TUEV data/output dirs, timeout, `num_ctx`, reasoning |
| `config/config.json` | `fs`, bandpass/notch, similarity threshold, channel-layout and age priors |

Current planner defaults (also in `.env.example`):

- Local: `OLLAMA_PLANNER_BASE_URL=http://127.0.0.1:11434/v1`, `OLLAMA_PLANNER_MODEL=qwen3.8:27b`
- Embeddings stay local: `OLLAMA_EMBED_MODEL=bge-m3:latest` at `http://127.0.0.1:11434/v1`
- TUEV: `TUEV_DATA_DIR` and `TUEV_OUT_DIR=runs/tuev_agent_ollama`

## Switch the planner

Local Qwen3.8 (default):

```
OLLAMA_PLANNER_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_PLANNER_API_KEY=ollama
OLLAMA_PLANNER_MODEL=qwen3.8:27b
```

Ollama Cloud (MiniMax or a cloud Qwen tag): set `OLLAMA_PLANNER_BASE_URL` to `https://ollama.com/v1`, put the cloud key in `OLLAMA_PLANNER_API_KEY` or `OLLAMA_API_KEY`, and set `OLLAMA_PLANNER_MODEL` to the cloud tag.

Leave `OLLAMA_REASONING_EFFORT=none` unless you want the planner to keep think blocks.

## Tests (no TUEV loop)

Parser only (no network):

```
python scripts/test_parse_calling.py
```

Smoke tests use `--phase`. Default is `env,planner,embed,rag,tools` (no full agent call).

```
python scripts/smoke_test.py
python scripts/smoke_test.py --phase planner
python scripts/smoke_test.py --phase embed
python scripts/smoke_test.py --phase env,planner,embed,rag,tools
python scripts/smoke_test.py --phase all
```

`--phase all` includes one agent turn on `data/gped_049_a_6.edf`.

## Evaluate

List TUEV eval pairs without calling the model:

```
python TUEV_eval.py --mode list
```

Two-file smoke eval (writes under `runs/`, gitignored):

```
python TUEV_eval.py --limit 2
```

Named files:

```
python TUEV_eval.py --file-list bckg_000_a_,gped_010_a_1 --out-dir runs/tuev_agent_ollama
```

Full official eval split (uses `TUEV_DATA_DIR` / `TUEV_OUT_DIR`):

```
python TUEV_eval.py
```

`--resume` skips windows already stored in that run’s `agent_raw.jsonl`.

Tool-only oracle (no LLM; same event-level score as the agent):

```
python TUEV_oracle_run.py --data-dir D:\Datasets\TUH-EEG\TUEV\v2.0.1\edf\eval --out-dir runs/tuev_oracle_run
```

Other eval scripts (need their local eval data under `eval/`):

```
python Sleep_eval.py
python MDD_eval.py
```

## Outputs

Eval writes generated logs and scores under `runs/`. That directory is gitignored and is recreated on the next eval. Do not treat `runs/` as source of truth for the codebase.
