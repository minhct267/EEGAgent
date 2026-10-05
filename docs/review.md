# Review

Two independent reads of [baseline_results.md](baseline_results.md) only: one of the experiment, one of the clinical meaning. This note keeps the points both the ledger and those reads support.

Three planners appear in the tables. They are not the same run.

| Name used below | Planner | Where the numbers are |
| --- | --- | --- |
| Qwen3.8:27B (local Ollama) | `qwen3.8:27b` through Ollama `/api/chat`, protocol `tuev_authors_v2` | `runs/tuev_authors_v2` (159 files, seed 0). The 36-file slice of that seed is `runs/rescore/e2_on_authors36`. Seeds 1 and 2, and all four ablations, use this same planner |
| Qwen3-235B (DashScope) | `qwen3-235b-a22b` on the DashScope API, git `acd2e6a` | `runs/rescore/acd2e6a`. 36 files only. This tree does not call DashScope |
| Qwen3.8:27B (earlier local folder) | The same `qwen3.8:27b` tag, but the strict 0.50 question, stored without a manifest | `runs/tuev_agent_ollama` |

The paper's 69.30% / 44.77% sentence names Qwen3-235B and does not name a file count. It is not either of the logged runs.

## What was measured

Qwen3.8:27B (local Ollama), seed 0, was asked about intervals already taken from `.rec` annotations, then scored offline. Coverage is 1,865/2,736 = 68.17%. IoU > 0.7 is 800/2,736 = 29.24%. Those are different rules. Extra predicted time does not lower coverage, so a long report on a short event can pass coverage and fail IoU.

On the same 2,736 events, Qwen3.8:27B (earlier local folder) scores 1,422/2,736. Of the 443-event coverage gap between that folder and the current Ollama protocol, SPSW contributes −1 hit (111 versus 112). GPED contributes 327 and PLED 117. The gap is real under this scorer. The earlier folder has no manifest, so its API, context, tool list, and harness are not recorded. Both sides of this gap are `qwen3.8:27b`. Neither side is DashScope.

On the 36 files that also have a Qwen3-235B (DashScope) log, Qwen3.8:27B (local Ollama) seed 0 is 334/447 coverage and 137/447 IoU. DashScope is 330/447 and 146/447. Ollama seeds 1 and 2 are 305 and 331 coverage hits. Ollama IoU hits stay at 137, 138, and 135. Ollama PLED hits move from 117 to 89 to 109 and account for most of that coverage spread. Ollama GPED stays at 191–195, next to DashScope's 190. Ollama SPSW stays at 24–27, under DashScope's 33. The 36-file bootstrap intervals are about 20 points wide and overlap. Ollama seed 0 of the full run is git `6eed8ef`, dirty. Ollama seeds 1 and 2 are `9a81722`, clean. This comparison does not isolate parameter count. The 1-second tools also score different rows than they did for the DashScope commit `acd2e6a` whenever the requested channels are not a prefix of the montage.

The same Ollama seed-0 run is a different operating point on 159 files than on 36: event F1 0.571 versus 0.500, and PLED coverage 491/887 versus 117/167. The 36-file pair is not a stand-in for the full Ollama split, and it is the only split that has a DashScope log.

## Oracle

The oracle calls one 1-second tool. It does not call Qwen3.8:27B or Qwen3-235B. It uses the same windows and the same 2,736 events as the full Ollama run. Pooled coverage of Qwen3.8:27B (local Ollama, seed 0) is below all four rules. The gaps are −159, −5, −187, and −56 hits against `seizNormal` at 0.5 and 0.7 and `seizArtiBckg` at 0.5 and 0.7. The five-hit gap against `seizNormal` at 0.7 is not a class-wise tie: SPSW is 111 versus 117, GPED 1,263 versus 1,271, and PLED 491 versus 482. Ollama IoU, 800, is below every oracle IoU. The nearest oracle count is 814. These are differences of pooled rates, not a paired test, and the oracle files do not store intervals.

The authors' question, the one Qwen3.8:27B (local Ollama) was given, does not tell that planner to threshold `seiz` at 0.5 or 0.7. The oracle is the ceiling of that rule, not of every way Qwen3.8:27B might read the tools. No oracle number exists for the DashScope planner.

## Ablations

Every ablation below is Qwen3.8:27B (local Ollama) on the 36-file list, seed 0. None of them is a DashScope run.

`authors_v1` returns 51 coverage hits. Of 120 windows, 13 have tuples, 93 are unparseable, and 14 are empty. That is the cost of the original XML splice on Qwen3.8:27B, under the authors' question. It is not the Qwen3-235B (DashScope) score of 330/447, and it is not the earlier local folder's 260/447.

`--prompt strict` changes the question, the tool list, and the `<NOTE>` lines together, still on Qwen3.8:27B. Coverage falls from 334 to 300 hits. The class mix moves the other way from a uniform drop: SPSW 26 to 37, GPED 191 to 157. One seed on 50 SPSW events is a thin estimate. The coverage interval still overlaps this planner's seed-0 range of 64.7–84.4.

`--rag off` on Qwen3.8:27B moves coverage from 334 to 311 and IoU from 137 to 143. GPED falls from 191 to 169. SPSW rises from 26 to 31. The coverage interval overlaps the seed range. One seed does not separate retrieval text from sampling noise.

`--think medium` on Qwen3.8:27B set `reasoning_effort` to `medium` and then lost 64 of 120 windows to `ReadTimeout`. The stored 74 hits (SPSW 11, GPED 34, PLED 29) are what that unfinished Ollama run produced. They are not an effect of reasoning, and they are not a DashScope result.

## Clinical reading

The class definitions below are properties of the TUEV labels, so they apply to every planner in the table. SPSW is a spike or sharp wave. GPED and PLED are periodic epileptiform discharges, generalized or lateralized. All three are epileptiform labels. An electrographic seizure, in the usual clinical reading, is a longer pattern: discharges faster than about 2.5 Hz for at least 10 seconds, or a pattern that evolves and lasts at least 10 seconds. A scored SPSW, GPED, or PLED event can be present when that definition is not. Of 2,736 events on the full split, 1,633 are GPED, so the pooled Qwen3.8:27B rate is mostly a periodic-discharge result. The question text, for both the Ollama protocol and the DashScope log, says "epileptic seizures", and the 1-second tools emit a class named `seiz`.

The windows already contain the annotated intervals. A hit is overlap with an event the question named in time. Neither the Ollama logs nor the DashScope log measure onset search, latency, false alarms per hour of unmarked EEG, or silence between annotations. There is no person, stimulator, or feedback that changes the recording. Neither planner run is a closed-loop BCI experiment.

The least ambiguous false-report counts are the reports on files with no positive event: 487 reports on 90 files for Qwen3.8:27B (local Ollama, seed 0), 128 reports on 18 files for Qwen3-235B (DashScope), and 308 reports on 90 files for Qwen3.8:27B (earlier local folder). The unmatched-report rate is a different quantity. For the current Ollama protocol it is 2,173/4,268 = 50.91% of merged reports. For DashScope on 36 files it is 506/814 = 62.16%. Neither figure has the paper's unstated false-rate denominator, so neither can be ranked against the paper's 44.77%. The IoU rates, about 29–33% for both logged planners, are hits over ground-truth events. They sit next to the paper's 69.30% hit-rate sentence, and they do not match it.

## Claims the tables do not support

- Qwen3.8:27B (local Ollama) coverage of 68.17% reproduces the paper's 69.30% for Qwen3-235B. One is coverage on 159 files. The other is an IoU sentence with no file count. The counterfactual 1,888/2,736 = 69.01% only says what Ollama coverage would be if the 23 events in its three timed-out windows all became hits.
- Qwen3.8:27B equals Qwen3-235B, or this design isolates parameter count. On the shared 36 files the coverage counts are close (334 versus 330 at seed 0) and the intervals overlap. SPSW does not match (24–27 versus 33). The tools, harness, and quantization also differ.
- Qwen3.8:27B (local Ollama) beats the 1-second detector. At 0.5 the detector is ahead by 159 or 187 hits. At 0.7 the pooled coverage gap is 5 or 56 hits, and IoU does not favor this planner. No such comparison exists for Qwen3-235B (DashScope).
- `--think medium` shows that reasoning lowers Qwen3.8:27B coverage. That Ollama run died on timeouts.
- 11.41% under `authors_v1` is the paper, or the Qwen3-235B (DashScope) run. It is Qwen3.8:27B on the original splice.
