# TUEV baseline

Protocol `tuev_authors_v2`. Planner `qwen3.8:27b` on local Ollama. This is the run to compare later planners against. It is not the paper's 69.30% / 44.77%, and it is not `runs/tuev_agent_ollama`.

How the three pipelines differ: [pipelines.md](pipelines.md). Full tables: [baseline_results.md](baseline_results.md).

## Frozen settings

From [config/protocols/tuev_authors_v2.json](../config/protocols/tuev_authors_v2.json) and the seed-0 manifest in `runs/tuev_authors_v2/manifest.json`.

| Piece | Setting |
| --- | --- |
| Question | Authors' text. Times are `round(start)` and `round(end)` |
| Tools | Every registered tool. The prompt shows the `acd2e6a` schemas |
| Tool results | `<FUNCTION>` / `<ARGS>` / `<RETURN>` / `<RESULT>` as a user turn. No `<NOTE>` |
| Harness | `authors_v2`. Stop `["<RETURN>"]`. One empty retry. One tool-free turn after round 8. Bad calls are returned as errors |
| Planner | `qwen3.8:27b`, Q4_K_M, family `qwen35`, 27.3B. API `ollama_native`. Digest `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643` |
| Sampling | temperature 0.7, top_p 0.8, top_k 20, seed 0, `reasoning_effort` `none` |
| Context | `num_ctx` 65536. The run loaded 65536. Overflow and truncation lists are empty |
| Retrieval | `bge-m3:latest`, F16, top 3, cosine at least 0.6 |
| Scorer | Coverage at 0.7, plus IoU > 0.7, per-class counts, event F1, zero-GT reports, file-level bootstrap (1,000 draws, seed 0) |
| Git | Full split, seed 0: `6eed8ef`, dirty. Seeds 1–2 and the four ablations: `9a81722`, clean |

`runs/tuev_authors_v2/run_health.json`: 512 windows, 3 errors, all `ReadTimeout`, loaded context 65536, `planner_api` `ollama_native`. The three failed windows are empty answers, so they add no hits. The 23 positive events inside those windows are current misses (18 class 2 on `gped_052_a_`, 2 class 2 on `spsw_023_a_1`, 3 class 3 on `pled_006_a_2`). If a rerun hit all 23 and left every other event unchanged, coverage would be 1,888/2,736 = 69.01%.

## Headline

Coverage is the scorer in the released code: same channel, merged overlaps cover at least 70% of the event. Extra predicted time is not penalized. IoU > 0.7 is the best same-channel report. The paper's sentence uses IoU. These two columns are not interchangeable.

| Source | Files | Coverage | IoU > 0.7 |
| --- | --- | --- | --- |
| Paper text. Planner named there: Qwen3-235B. File count not stated | not stated | not stated | 69.30% hit rate; false rate 44.77%, denominator not stated |
| Qwen3-235B, DashScope API, git `acd2e6a` | 36 | 330/447 = 73.83% (60.4–83.9) | 146/447 = 32.66% (19.4–44.6) |
| Qwen3.8:27B, local Ollama, `tuev_authors_v2`, seed 0, same 36 files | 36 | 334/447 = 74.72% (64.7–84.4) | 137/447 = 30.65% (19.0–42.3) |
| Qwen3.8:27B, local Ollama, `tuev_authors_v2`, seed 0, full eval split | 159 | 1,865/2,736 = 68.17% (59.7–74.5) | 800/2,736 = 29.24% (22.7–35.5) |
| Qwen3.8:27B, earlier local folder `tuev_agent_ollama`, strict 0.50 question | 159 | 1,422/2,736 = 51.97% (43.0–59.9) | 692/2,736 = 25.29% (19.6–29.2) |
| Qwen3.8:27B, that same earlier folder, 36-file subset | 36 | 260/447 = 58.17% (45.5–68.9) | 122/447 = 27.29% (15.9–38.2) |

Intervals are file-level bootstrap 95% ranges. Every `metrics_v2.json` behind this table has `sanity.coverage_hits_match_stored` true.

68.17% is Qwen3.8:27B (local Ollama) coverage on 159 files. It is not the paper's IoU hit rate for Qwen3-235B. On the 36 shared files, Ollama seed-0 coverage is 334 against DashScope's 330, and IoU hits are 137 against 146. Ollama seeds 1 and 2 on those files are 68.23% and 74.05% coverage. That spread, plus the channel-index fix, the `Eye movement` key, harness `authors_v2`, local Q4_K_M versus the DashScope API model, and BGE-M3 versus the paper's Qwen3-Embedding-8B sentence, blocks reading the table as a parameter-count result.

## Reproduce

From the repo root, with `.env` copied from `.env.example` and `OLLAMA_PLANNER_API=ollama_native`. `TUEV_DATA_DIR` is the TUEV v2.0.1 eval split: 159 pairs, 512 windows. The conda env used for these runs is `/data/nmduong/cache/conda/envs/bci`.

These directories already exist. The commands are how they were produced. `--sleep` defaults to 5 seconds per window; the commands pass `--sleep 0`. Do not run two `TUEV_eval.py` processes at once. They share one Ollama server.

```bash
python scripts/test_parse_calling.py
python scripts/test_tuev_metrics.py
python scripts/test_harness.py
python scripts/baseline_smoke.py

python TUEV_oracle_run.py --rule both --threshold 0.5 --threshold 0.7 \
    --out-dir runs/tuev_oracle

python TUEV_eval.py --protocol tuev_authors_v2 --seed 0 --sleep 0 \
    --out-dir runs/tuev_authors_v2

for s in 1 2; do
  python TUEV_eval.py --protocol tuev_authors_v2 --seed "$s" --sleep 0 \
      --file-list config/protocols/authors_235b_files.txt \
      --out-dir "runs/tuev_authors_v2_36_seed${s}"
done

F=config/protocols/authors_235b_files.txt
python TUEV_eval.py --harness authors_v1 --sleep 0 --file-list "$F" --out-dir runs/tuev_ablate_harness_v1
python TUEV_eval.py --prompt strict      --sleep 0 --file-list "$F" --out-dir runs/tuev_ablate_strict
python TUEV_eval.py --rag off            --sleep 0 --file-list "$F" --out-dir runs/tuev_ablate_rag_off
python TUEV_eval.py --think medium       --sleep 0 --file-list "$F" --out-dir runs/tuev_ablate_think_medium
```

`--resume` skips windows already in `agent_raw.jsonl` whose transcript was saved. A raw line with no transcript and no error is run again. A window that failed with an exception is kept.

Rescore, after a run:

```bash
python scripts/rescore_tuev.py --run-dir runs/tuev_authors_v2 \
    --out runs/tuev_authors_v2/metrics_v2.json
python scripts/rescore_tuev.py --run-dir runs/tuev_authors_v2 --stems-from-git acd2e6a \
    --out runs/rescore/e2_on_authors36/metrics_v2.json
python scripts/rescore_tuev.py --git acd2e6a --git-prefix runs/tuev_agent \
    --out runs/rescore/acd2e6a/metrics_v2.json
```

Expect `files=159` and `total_gt=2736` on the full Qwen3.8:27B split, and `files=36` and `total_gt=447` on the 36-file outputs. The Qwen3-235B (DashScope) rescore is `hits=330`, `total_gt=447`, `iou_hits=146`.
