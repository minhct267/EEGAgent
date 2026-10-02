# Running the Qwen3.8 TUEV baseline

Commands for the frozen protocol `tuev_authors_v2`. What that protocol freezes, and how it differs from the authors' 235B run, is in [baseline.md](baseline.md). This file is only the procedure: exact commands, what each run is for, and the numbers to check the output against.

Run everything from the repo root with the conda env that has the project dependencies:

```bash
cd /home/nmduong/Data/projects/eeg-agent
conda activate /data/nmduong/cache/conda/envs/bci
```

`python` below means that env (`/data/nmduong/cache/conda/envs/bci/bin/python`). `.env` must exist (copy `.env.example`). Do not commit `.env`. `TUEV_DATA_DIR` in `.env` is the official TUEV v2.0.1 eval split: 159 paired `.edf` / `.rec` files, 512 windows. `config/protocols/authors_235b_files.txt` is the 36 files (120 windows) in the authors' released run, git `acd2e6a:runs/tuev_agent`.

`OLLAMA_PLANNER_API` must be `ollama_native`. That calls Ollama's own `/api/chat` on the same local server (`127.0.0.1:11434`). Ollama's OpenAI-compatible `/v1` ignores `num_ctx` and `top_k`, so a run that looks configured for 65,536 tokens would actually load the server default (32,768 on Ollama 0.35.0) and silently drop the oldest messages past that. `/api/chat` loads the requested context and, with `truncate: false`, returns an error instead of dropping messages. A window that does not fit is recorded as `context_overflow`.

## What to compare against

These numbers already exist. They are not the new baseline. The new baseline is E2 below, and it has not been run yet.

| Reference | Files | Coverage | IoU > 0.7 | Where |
| --- | --- | --- | --- | --- |
| Authors' `qwen3-235b-a22b`, git `acd2e6a` | 36 | 330/447 = 73.83% (95% CI 60.4–83.9) | 146/447 = 32.66% | `runs/rescore/acd2e6a/metrics_v2.json` |
| Old local `qwen3.8:27b`, same 36 files | 36 | 260/447 = 58.17% (95% CI 45.5–68.9) | 122/447 = 27.29% | `runs/rescore/local_on_authors36/metrics_v2.json` |
| Old local `qwen3.8:27b`, full split | 159 | 1,422/2,736 = 51.97% (95% CI 43.0–59.9) | 692/2,736 = 25.29% | `runs/tuev_agent_ollama/metrics_v2.json` |

The two local rows used a different question, nine tools, model-name patches, and a context the `/v1` client did not actually apply. Do not treat 51.97% or 58.17% as the score this protocol should reproduce.

The paper's 69.30% hit rate and 44.77% false rate are also not the target. The released scorer is coverage at overlap 0.7, not IoU, and the released 235B run is 36 files, not 159. Its coverage is 73.83%.

Per-class coverage on the paired 36 files (447 events):

| Class | Events | 235B | Old local qwen3.8 |
| --- | --- | --- | --- |
| SPSW | 50 | 33 (66.0%) | 37 (74.0%) |
| GPED | 230 | 190 (82.6%) | 146 (63.5%) |
| PLED | 167 | 107 (64.1%) | 77 (46.1%) |

Headline comparisons once E1 and E2 finish:

- E2 restricted to the 36 files, against the 235B row (same files, same 447 events, same scorer).
- E2 on all 159 files, against the E1 oracle. The oracle is the detector ceiling when a fixed rule reads the 1-second tools. The authors' question does not give the planner that rule, so the planner can report events the oracle would drop. Read the gap as planner behavior, not as a hard upper bound.

State the remaining differences next to any 235B comparison. They are listed in [baseline.md](baseline.md): the channel-index fix in the three 1-second tools, the `Eye movement` key, harness `authors_v2`, and local Q4_K_M `qwen3.8:27b` against DashScope `qwen3-235b-a22b`.

## Before the long runs

Commit the tree first if you want `manifest.json` to name the code that ran. `runs/` and `.env` are gitignored.

```bash
git status --short
git add -A
git commit -m "Freeze TUEV authors_v2 baseline"
git tag tuev-baseline-v2
```

Use a persistent session. Closing the terminal kills the eval.

```bash
tmux new -s eeg
```

Checks, about 5–10 minutes. Stop if any of them fails.

```bash
python scripts/test_parse_calling.py
python scripts/test_tuev_metrics.py
python scripts/test_harness.py
python scripts/smoke_test.py
python scripts/baseline_smoke.py
```

Expect the last line `Baseline smoke passed.` and, above it, `loaded_ctx=65536`. `stop_hits=0` is expected on Ollama: the server strips the stop string `<RETURN>` before the client sees it. The check that matters is that no assistant turn contains `<RETURN>`. The smoke files (`bckg_014_a_`, `bckg_024_a_`) have no positive labels, so their reports are not a score.

Confirm the split sizes without calling the model:

```bash
python TUEV_eval.py --mode list
python TUEV_oracle_run.py --mode list
```

Expect `Found 159 paired .rec/.edf files`.

## Order and time

E1 and E2 are the baseline. E3 measures seed spread. E4 is the analysis and can wait.

Do not run two `TUEV_eval.py` processes at once. They share one Ollama server, and each request needs the 65,536-token context (about 20.5 GB across the two L4s when measured on 2 Oct 2026). Do not also call `qwen3.8:27b` through `/v1` while an eval is running: that reloads the model at the server default and slows the run. The native client will load 65,536 again on its next call, so the scores stay valid, but the reload costs time.

Times below are estimates from the 2 Oct 2026 smoke (about 1.5–2.5 minutes per window), not a guarantee. `--think medium` is slower because the model also emits reasoning tokens.

| Step | Windows | Estimate |
| --- | --- | --- |
| E1 oracle, four rule/threshold pairs | 512, no LLM | 1–2 hours, CPU |
| E2 full baseline | 512 | 15–20 hours |
| E3, each extra seed | 120 | 4–5 hours |
| E4, each ablation | 120 | 4–5 hours; longer for `--think medium` |

`--sleep` defaults to 5 seconds per window. Every eval command below passes `--sleep 0`. Leaving it off adds about 40 minutes on E2 and about 10 minutes on a 120-window run.

E1 does not call Ollama. It can run in a second tmux window during E2. Both load the EEG tool weights into CPU memory, so the machine will be busier, but they do not share the planner context.

## E1 — oracle

Tool-only ceiling. Same rounded windows as the agent, same scorer. A second is kept when `seiz` is at least the threshold. Four combinations: `seizureNormalModel_OneSecond` and `seizureArtiBckgModel_OneSecond`, each at 0.5 and at 0.7.

```bash
python TUEV_oracle_run.py --rule both --threshold 0.5 --threshold 0.7 \
    --out-dir runs/tuev_oracle
```

Outputs, one directory per combination:

- `runs/tuev_oracle/seizNormal_t0.5/`
- `runs/tuev_oracle/seizNormal_t0.7/`
- `runs/tuev_oracle/seizArtiBckg_t0.5/`
- `runs/tuev_oracle/seizArtiBckg_t0.7/`

Read `summary.json` in each. There is no expected coverage yet; this run defines the ceiling. Expect `total_gt` of 2736 on the full split (447 if you pass `--file-list config/protocols/authors_235b_files.txt`).

## E2 — primary baseline

Authors' question, every tool, harness `authors_v2`, `qwen3.8:27b`, thinking off, RAG on, seed 0, `num_ctx` 65536.

```bash
python TUEV_eval.py --protocol tuev_authors_v2 --seed 0 --sleep 0 \
    --out-dir runs/tuev_authors_v2
```

The first progress line should contain all of these: `159 files / 512 candidate windows`, `protocol=tuev_authors_v2`, `harness=authors_v2`, `prompt=authors`, `rag=on`, `seed=0`, `think=none`, `model=qwen3.8:27b`, `api=ollama_native`, `num_ctx=65536`.

If the process stops, rerun the same command with `--resume` added. `--resume` skips windows already in `agent_raw.jsonl`. A window whose raw line has no transcript and no error is run again. A window that failed with an exception is kept and is not retried; delete that raw line and its row in `messages/<stem>.messages.jsonl` before resuming if you want it rerun.

When it finishes, `runs/tuev_authors_v2/run_health.json` should show:

| Field | Expect |
| --- | --- |
| `windows` | 512 |
| `errors` | 0 |
| `planner_api` | `ollama_native` |
| `num_ctx_requested` | 65536 |
| `ollama_loaded_context` | 65536 |
| `context_overflow_windows` | `[]` |
| `context_truncated_windows` | `[]` |

`manifest.json` should name protocol `tuev_authors_v2`, harness stop `["<RETURN>"]`, seed 0, planner `api` `ollama_native`, and a planner digest. `stop_hit` stays 0 on Ollama; that is not a failure.

If overflow or truncation is non-empty, report the score together with how many windows are listed. Those windows did not see the full conversation.

`format_status` counts are descriptive, not a pass/fail. The authors' question never says to write `No events found`, so a prose refusal is `unparseable` and contributes no events. That happened on one of the two smoke windows.

## E3 — two more seeds, 36 files

Seed 0 on these 36 files is already inside E2. Rescore extracts it (see the next section). Run only seeds 1 and 2.

```bash
for s in 1 2; do
  python TUEV_eval.py --protocol tuev_authors_v2 --seed "$s" --sleep 0 \
      --file-list config/protocols/authors_235b_files.txt \
      --out-dir "runs/tuev_authors_v2_36_seed${s}"
done
```

Expect `36 files / 120 candidate windows` and the same health checks as E2, with `windows` 120 and `sampling.seed` equal to 1 or 2. The spread across seeds 0, 1, and 2 is the sampling noise around the 36-file point estimate. The 235B interval (60.4–83.9) is the uncertainty to put next to that comparison; it is wide because it is 36 files.

## E4 — one factor at a time, 36 files

Each command changes one flag from E2. Sampling stays at seed 0 and thinking off unless the command says otherwise.

```bash
F=config/protocols/authors_235b_files.txt
python TUEV_eval.py --harness authors_v1 --sleep 0 --file-list "$F" --out-dir runs/tuev_ablate_harness_v1
python TUEV_eval.py --prompt strict      --sleep 0 --file-list "$F" --out-dir runs/tuev_ablate_strict
python TUEV_eval.py --rag off            --sleep 0 --file-list "$F" --out-dir runs/tuev_ablate_rag_off
python TUEV_eval.py --think medium       --sleep 0 --file-list "$F" --out-dir runs/tuev_ablate_think_medium
```

| Directory | What changed | What the difference from E2 (36 files, seed 0) means |
| --- | --- | --- |
| `tuev_ablate_harness_v1` | Results spliced into the assistant message. No stop sequence, no empty retry, no forced final turn. The authors' parser drops bad calls with no feedback. | Cost of the original XML splice on qwen3.8. The saved local run needed an extra "continue" user turn after every tool result; this ablation is the clean measurement. |
| `tuev_ablate_strict` | The 0.50 question, nine discharge tools, the rewritten tool descriptions, and `<NOTE>` lines after tool results. | Whether the old local prompt helps or hurts. It will not reproduce the old 58.17%, because that run also changed the harness, the context, and the tool text path. |
| `tuev_ablate_rag_off` | No retrieved chunks. | How much the clinical seizure definitions (often "usually >10 s") move the score on 1-second events. A higher coverage here means those chunks were suppressing short events. |
| `tuev_ablate_think_medium` | `think` sent as the string `medium` instead of `false`. | Whether reasoning changes coverage. Direction is not known in advance. This run is the slow one. |

Same health checks as E2, with `windows` 120. For the think run, `manifest.json` sampling `reasoning_effort` must be `medium`.

## Rescore

Scorer v2 adds IoU, per-class recall, event precision and F1, zero-GT reports, and a file-level bootstrap 95% interval. `TUEV_eval.py` already writes coverage into each file's `summary.json`. Rescore puts the v2 aggregate in one file and checks that coverage hits still match the stored predictions (`sanity: true`).

```bash
python scripts/rescore_tuev.py --run-dir runs/tuev_authors_v2 \
    --out runs/tuev_authors_v2/metrics_v2.json

python scripts/rescore_tuev.py --run-dir runs/tuev_authors_v2 --stems-from-git acd2e6a \
    --out runs/rescore/e2_on_authors36/metrics_v2.json

for d in tuev_authors_v2_36_seed1 tuev_authors_v2_36_seed2 \
         tuev_ablate_harness_v1 tuev_ablate_strict tuev_ablate_rag_off tuev_ablate_think_medium; do
  python scripts/rescore_tuev.py --run-dir "runs/$d" --out "runs/$d/metrics_v2.json"
done
```

Read `stored_predictions` in each `metrics_v2.json`. The fields to put in the comparison table are `hits`, `total_gt`, `hit_rate`, `iou_hits`, `iou_hit_rate`, `event_f1`, and the per-class `spsw_*`, `gped_*`, `pled_*` counts. The printed line is the same summary.

Checks that the file set is the one you think it is:

| Rescore | Expect |
| --- | --- |
| E2, all files | `files=159`, `total_gt=2736` |
| E2, `--stems-from-git acd2e6a` | `files=36`, `total_gt=447` |
| Each E3 and E4 directory | `files=36`, `total_gt=447` |
| Every file | `sanity.coverage_hits_match_stored` true |

`total_gt=447` on the 36-file outputs is the check that they cover the same events as the 235B row. Compare `hit_rate` with 0.7383 and `iou_hit_rate` with 0.3266.

The 235B rescore is already written. Regenerating it does not call a model:

```bash
python scripts/rescore_tuev.py --git acd2e6a --git-prefix runs/tuev_agent \
    --out runs/rescore/acd2e6a/metrics_v2.json
```

Expect `hits=330`, `total_gt=447`, `hit_rate` about 0.7383, `iou_hits=146`.
