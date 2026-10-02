# EEGAgent runbook

Commands assume the repo root and the conda env at `/data/nmduong/cache/conda/envs/bci` (`python` on that env). Copy `.env.example` to `.env` before the first run. Do not commit `.env`.

## Datasets

The code reads these paths. It does not write into them.

- TUEV eval split used by `TUEV_eval.py`: `/home/nmduong/Data/datasets/TUH-EEG/TUEV/v2.0.1/edf/eval` (`TUEV_DATA_DIR` in `.env`). That directory is the official eval split, 159 paired `.edf` / `.rec` files.
- TUAB is not set in `.env`. `load_MDD_edf` and `load_Sleep_edf` are commented out in `main.py`, so `Sleep_eval.py` and `MDD_eval.py` are not wired to the agent.

A small sample EDF for smoke tests lives at `data/gped_049_a_6.edf`.

## Config

| File | What it controls |
| --- | --- |
| `.env` | Planner URL, model, API, timeout, and the default sampling (`num_ctx` 65536, temperature 0.7, top_p 0.8, top_k 20, seed 0, reasoning `none`). `OLLAMA_PLANNER_API=ollama_native` calls `/api/chat`; Ollama's `/v1` ignores `num_ctx` and `top_k`. |
| `config/protocols/authors_tool_schemas.json` | The tool schemas registered at `acd2e6a`, shown by the authors prompt. Regenerate with `python scripts/export_authors_tool_schemas.py`. |
| `config/config.json` | Sample rate, bandpass, notch, RAG similarity threshold 0.6, channel and age priors |
| `config/protocols/tuev_authors_v2.json` | The frozen TUEV question, tool set, harness, and sampling. Eval flags override this file. The file overrides `.env` sampling. |

`TUEV_OUT_DIR` defaults to `runs/tuev_authors_v2`. The previous local run stays in `runs/tuev_agent_ollama`.

## Baseline command

The full procedure, with checks, oracle, seeds, ablations, rescore, and the numbers to compare against, is in [baseline_run.md](baseline_run.md). Pass `--sleep 0` on long evals; the default is 5 seconds per window.

Authors' question, all tools, harness `authors_v2` (stop at `<RETURN>`, tool results as a user turn, one empty retry, one forced final turn), planner `qwen3.8:27b`, RAG `bge-m3:latest`, seed 0:

```
python TUEV_eval.py --protocol tuev_authors_v2 --seed 0 --out-dir runs/tuev_authors_v2
```

Ablations on the same code:

```
python TUEV_eval.py --harness authors_v1 --file-list config/protocols/authors_235b_files.txt --out-dir runs/tuev_ablate_harness_v1
python TUEV_eval.py --prompt strict --file-list config/protocols/authors_235b_files.txt --out-dir runs/tuev_ablate_strict
python TUEV_eval.py --rag off --file-list config/protocols/authors_235b_files.txt --out-dir runs/tuev_ablate_rag_off
python TUEV_eval.py --think medium --file-list config/protocols/authors_235b_files.txt --out-dir runs/tuev_ablate_think_medium
```

`--harness authors_v1` is the original loop: results spliced into the assistant message, no stop sequence, no empty retry, no forced final turn, and the authors' parser, which skips bad calls without feedback. `--prompt strict` is the 0.50 question, the nine discharge tools, the rewritten tool text, and `<NOTE>` lines after tool results. Sampling stays the protocol sampler unless `--seed` or `--think` is set.

`--resume` skips windows already in `agent_raw.jsonl` and reloads transcripts from `messages/<stem>.messages.jsonl`. A window whose transcript was not saved is run again.

## Tests

Parser, scorer, and harness (no TUEV files):

```
python scripts/test_parse_calling.py
python scripts/test_tuev_metrics.py
python scripts/test_harness.py
```

Two-file live check (manifest, stop sequence, native API and loaded context, authors' tool text, no `<NOTE>`, resume, token counts):

```
python scripts/baseline_smoke.py
```

Component smoke, without the TUEV loop:

```
python scripts/smoke_test.py
python scripts/smoke_test.py --phase all
```

List pairs without calling the model:

```
python TUEV_eval.py --mode list
```

## Oracle

Same windows as the agent (`round` of each merged annotation span) and the same scorer. A second is kept when `seiz` is at least the threshold. Run both tools at 0.5 and at 0.7:

```
python TUEV_oracle_run.py --rule both --threshold 0.5 --threshold 0.7 --out-dir runs/tuev_oracle
```

One tool at the authors' old default:

```
python TUEV_oracle_run.py --rule seizNormal --threshold 0.7 --out-dir runs/tuev_oracle_seizNormal_t0.7
```

## Re-score a saved run

```
python scripts/rescore_tuev.py --run-dir runs/tuev_agent_ollama --out runs/tuev_agent_ollama/metrics_v2.json
python scripts/rescore_tuev.py --git acd2e6a --git-prefix runs/tuev_agent --out runs/rescore/acd2e6a/metrics_v2.json
```

## Outputs

Each run directory gets `manifest.json` (git commit, protocol, prompt hash, planner API, model digest, FAISS, chunk, and authors' tool-schema hashes, file list), `run_health.json` (answer status counts, harness events, largest prompt, loaded context, overflowed or truncated windows), and per-file `agent_raw.jsonl` plus `messages/<stem>.messages.jsonl`. `runs/` is gitignored.
