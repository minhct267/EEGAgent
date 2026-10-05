# Baseline results

Numbers below were read from `runs/` on 5 Oct 2026. Every `metrics_v2.json` has `sanity.coverage_hits_match_stored` true. Rates are `hits / total_gt`. Bootstrap intervals are the file-level 95% ranges stored in the same files (1,000 draws, seed 0).

Protocol and commands: [baseline.md](baseline.md). What each run changed: [pipelines.md](pipelines.md).

## Metric definitions

| Column | Definition in `utils/tuev_metrics.py` |
| --- | --- |
| Coverage | A positive event is a hit when the union of same-channel report overlaps covers at least 70% of its duration. Extra predicted time is not penalized |
| IoU > 0.7 | The best same-channel report has intersection-over-union above 0.7 |
| Unmatched reports | Merged reports with no positive overlap, divided by all merged reports |
| Event F1 | Harmonic mean of event precision (reports that overlap any positive event) and coverage recall |
| Zero-GT reports | Merged reports on files whose `total_gt` is 0 |

Positive classes are 1 SPSW, 2 GPED, 3 PLED (`TUEV/v2.0.1/AAREADME.txt`). Filename prefixes are not the class field. The two events inside the `spsw_023_a_1` timeout window are class 2.

The paper states hit rate 69.30% and false rate 44.77% under IoU > 0.7. It does not state the file count or the false-rate denominator. Coverage is a different rule.

## Full split and the 36-file pair

| Run | Files | Coverage | IoU > 0.7 | Unmatched reports | Event F1 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8:27B, local Ollama, `tuev_authors_v2`, seed 0 | 159 | 1,865/2,736 = 68.17% (59.7–74.5) | 800/2,736 = 29.24% (22.7–35.5) | 2,173/4,268 = 50.91% | 0.571 |
| Same Qwen3.8:27B run, 36-file subset | 36 | 334/447 = 74.72% (64.7–84.4) | 137/447 = 30.65% (19.0–42.3) | 525/841 = 62.43% | 0.500 |
| Qwen3-235B, DashScope, git `acd2e6a` | 36 | 330/447 = 73.83% (60.4–83.9) | 146/447 = 32.66% (19.4–44.6) | 506/814 = 62.16% | 0.500 |
| Qwen3.8:27B, earlier local folder `tuev_agent_ollama` | 159 | 1,422/2,736 = 51.97% (43.0–59.9) | 692/2,736 = 25.29% (19.6–29.2) | 1,734/3,418 = 50.73% | 0.506 |
| Qwen3.8:27B, that earlier folder, 36-file subset | 36 | 260/447 = 58.17% (45.5–68.9) | 122/447 = 27.29% (15.9–38.2) | 423/694 = 60.95% | 0.467 |

Qwen3.8:27B (local Ollama, seed 0) has 90 files with no positive event and 487 reports on those files. Qwen3-235B (DashScope, `runs/rescore/acd2e6a`) has 18 such files and 128 reports. Qwen3.8:27B (earlier folder `tuev_agent_ollama`) has 90 such files and 308 reports.

## Health of Qwen3.8:27B, local Ollama, seed 0

`runs/tuev_authors_v2/run_health.json` and `manifest.json`:

| Field | Value |
| --- | --- |
| Windows | 512 |
| Errors | 3, each `ReadTimeout` |
| Answer status | tuples 396, unparseable 113, empty 3 |
| Loaded context | 65536, requested 65536, API `ollama_native` |
| Overflow, truncation | both empty |
| Largest prompt | 22,326 tokens |
| Git | `6eed8ef051ff0c346d1357edee8683638ac62491`, dirty |
| Embedder | `bge-m3:latest`, F16, 566.70M |

The empty windows are `gped_052_a_` candidate 1 (202.9–211.2 s, 18 class-2 events), `spsw_023_a_1` candidate 1 (39.2–41.4 s, 2 class-2 events), and `pled_006_a_2` candidate 1 (22.5–32.2 s, 3 class-3 events). All 23 are current misses. 1,865 + 23 = 1,888, and 1,888/2,736 = 69.01%, if a rerun of only those windows hit every one of them and left the other events unchanged.

## Per class, full split

Counts sum to the coverage numerator.

| Class | Events | Qwen3.8:27B, local Ollama, seed 0 | Oracle, no planner, `seizNormal` 0.5 | Oracle, no planner, `seizNormal` 0.7 | Qwen3.8:27B, earlier folder |
| --- | --- | --- | --- | --- | --- |
| SPSW | 216 | 111 (51.39%) | 143 (66.20%) | 117 (54.17%) | 112 (51.85%) |
| GPED | 1,633 | 1,263 (77.34%) | 1,343 (82.24%) | 1,271 (77.83%) | 936 (57.32%) |
| PLED | 887 | 491 (55.36%) | 538 (60.65%) | 482 (54.34%) | 374 (42.16%) |

## Oracle

`runs/tuev_oracle/*/summary.json`. 159 files, 2,736 events. No LLM.

| Rule | Threshold | Coverage | IoU > 0.7 | SPSW | GPED | PLED |
| --- | --- | --- | --- | --- | --- | --- |
| `seizNormal` | 0.5 | 2,024/2,736 = 73.98% | 844/2,736 = 30.85% | 143/216 = 66.20% | 1,343/1,633 = 82.24% | 538/887 = 60.65% |
| `seizNormal` | 0.7 | 1,870/2,736 = 68.35% | 814/2,736 = 29.75% | 117/216 = 54.17% | 1,271/1,633 = 77.83% | 482/887 = 54.34% |
| `seizArtiBckg` | 0.5 | 2,052/2,736 = 75.00% | 889/2,736 = 32.49% | 158/216 = 73.15% | 1,338/1,633 = 81.94% | 556/887 = 62.68% |
| `seizArtiBckg` | 0.7 | 1,921/2,736 = 70.21% | 864/2,736 = 31.58% | 137/216 = 63.43% | 1,279/1,633 = 78.32% | 505/887 = 56.93% |
| Qwen3.8:27B, local Ollama, seed 0 | — | 1,865/2,736 = 68.17% | 800/2,736 = 29.24% | 111/216 = 51.39% | 1,263/1,633 = 77.34% | 491/887 = 55.36% |

Pooled coverage of Qwen3.8:27B (local Ollama, seed 0) minus pooled coverage of each oracle. Same denominator, 2,736. The oracle does not call either planner. No oracle row exists for Qwen3-235B (DashScope).

| Oracle | Hit difference | Rate difference |
| --- | --- | --- |
| `seizNormal` 0.5 | 1,865 − 2,024 = −159 | −5.81 points |
| `seizNormal` 0.7 | 1,865 − 1,870 = −5 | −0.18 points |
| `seizArtiBckg` 0.5 | 1,865 − 2,052 = −187 | −6.83 points |
| `seizArtiBckg` 0.7 | 1,865 − 1,921 = −56 | −2.05 points |

These are exact hit differences over 2,736. They are not the difference of the rounded percentages, and they are not a paired bootstrap. `-56/2736` is −2.05 points; 70.21 − 68.17 is 2.04 because both rates were already rounded.

## Seeds on the 36 files

447 events: SPSW 50, GPED 230, PLED 167. Seeds 0, 1, and 2 are Qwen3.8:27B on local Ollama (`authors_v2`, RAG on, `think: false`). Seed 0 is the 36-file slice of the full Ollama run. The last row is Qwen3-235B on DashScope, not another seed of Qwen3.8:27B.

| Seed | Coverage | IoU > 0.7 | SPSW | GPED | PLED | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 334/447 = 74.72% (64.7–84.4) | 137/447 = 30.65% (19.0–42.3) | 26 | 191 | 117 | 0 on this subset |
| 1 | 305/447 = 68.23% (56.2–79.3) | 138/447 = 30.87% (19.2–41.4) | 24 | 192 | 89 | 4 |
| 2 | 331/447 = 74.05% (63.4–83.9) | 135/447 = 30.20% (20.0–41.0) | 27 | 195 | 109 | 1 |
| Qwen3-235B, DashScope | 330/447 = 73.83% (60.4–83.9) | 146/447 = 32.66% (19.4–44.6) | 33 | 190 | 107 | not in an Ollama health file |

Seed 1 status: tuples 85, unparseable 31, empty 4. Seed 2: tuples 82, unparseable 37, empty 1. Both loaded context 65536, no overflow, no truncation. Git `9a81722`, clean. Event F1 is 0.506 (seed 1) and 0.511 (seed 2).

For Qwen3.8:27B (local Ollama), seed order 0, 1, 2, the coverage counts are 334, 305, and 331. SPSW hits are 26, 24, and 27. GPED hits are 191, 192, and 195. PLED hits are 117, 89, and 109. Qwen3-235B (DashScope) has SPSW 33, GPED 190, and PLED 107.

## Ablations, 36 files, seed 0

Each directory is Qwen3.8:27B on local Ollama, one flag away from `tuev_authors_v2`. None of them calls DashScope. Health: 120 windows, context 65536, no overflow, no truncation, git `9a81722` clean.

| Run | Coverage | IoU > 0.7 | SPSW | GPED | PLED | Errors | Answer status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8:27B, seed 0, 36 files | 334/447 = 74.72% | 137/447 = 30.65% | 26 | 191 | 117 | 0 on this subset | see the full Ollama run |
| `authors_v1` | 51/447 = 11.41% (2.9–18.6) | 27/447 = 6.04% (0.0–13.7) | 11 | 19 | 21 | 14 | tuples 13, unparseable 93, empty 14 |
| `--prompt strict` | 300/447 = 67.11% (54.8–81.9) | 120/447 = 26.85% (16.8–38.7) | 37 | 157 | 106 | 1 | tuples 76, no_events 43, empty 1 |
| `--rag off` | 311/447 = 69.57% (58.3–80.7) | 143/447 = 31.99% (21.3–42.7) | 31 | 169 | 111 | 4 | tuples 77, unparseable 39, empty 4 |
| `--think medium` | 74/447 = 16.55% (8.7–23.6) | 35/447 = 7.83% (1.5–15.7) | 11 | 34 | 29 | 64 | tuples 31, unparseable 25, empty 64 |

`authors_v1` manifest: harness `authors_v1`, stop `[]`, prompt `authors`. Event F1 0.190.

`--prompt strict` manifest: prompt `strict`, and `tool_names` is the nine `DISCHARGE_TOOL_NAMES`. Event F1 0.488. Class hits: SPSW 37, GPED 157, PLED 106.

`--rag off` manifest: `rag.enabled` false. Event F1 0.497. Class hits: SPSW 31, GPED 169, PLED 111.

`--think medium` manifest: `reasoning_effort` `medium`. All 64 errors are `ReadTimeout`. Class hits still sum to 74 (11 + 34 + 29). Event F1 0.240. Largest prompt in that directory is 13,325 tokens.

## Sources

| Number | File |
| --- | --- |
| Full baseline | `runs/tuev_authors_v2/metrics_v2.json`, `run_health.json`, `manifest.json` |
| 36-file seed 0 | `runs/rescore/e2_on_authors36/metrics_v2.json` |
| Seeds 1 and 2 | `runs/tuev_authors_v2_36_seed1/`, `runs/tuev_authors_v2_36_seed2/` |
| Ablations | `runs/tuev_ablate_harness_v1/`, `tuev_ablate_strict/`, `tuev_ablate_rag_off/`, `tuev_ablate_think_medium/` |
| Oracle | `runs/tuev_oracle/*/summary.json` |
| Qwen3-235B, DashScope | `runs/rescore/acd2e6a/metrics_v2.json` |
| Qwen3.8:27B, earlier local folder | `runs/tuev_agent_ollama/metrics_v2.json`, `runs/rescore/local_on_authors36/metrics_v2.json` |
