# TUEV run

Source: `runs/tuev_agent_ollama`, recomputed from `aggregate_summary.json` and each `agent_predictions.jsonl`. Planner logged on all 512 windows: `qwen3.8:27b`. The embedding model is not logged. `.env` sets `bge-m3:latest`. Data directory: `/home/nmduong/Data/datasets/TUH-EEG/TUEV/v2.0.1/edf/eval` (159 EDF/REC pairs, the official eval split in `AAREADME.txt`).

The authors' 235B run is in git `acd2e6a:runs/tuev_agent` (36 files, 120 windows), not in `runs/`. On those same 36 files this local run scores 58.17% coverage against their 73.83%. Both numbers use the repository scorer. There is still no tool-only oracle in `runs/`. The paired table is in [baseline.md](baseline.md).

## Paper result

On the TUEV test set the paper merges events less than 1 s apart, treats SPSW, GPED, and PLED as the positive classes, and counts a prediction as correct when its IoU with the reference on the same channel exceeds 0.7. Reported hit rate: **69.30%**. Reported false rate: **44.77%**. The paper does not define the false-rate denominator, the number of files, or how windows were chosen.

## Local aggregate

Positive events are `.rec` classes 1–3 after merging same channel and same class across gaps of at most 1 s. Official codes (`TUEV/v2.0.1/AAREADME.txt`): 1 SPSW, 2 GPED, 3 PLED, 4 EYEM, 5 ARTF, 6 BCKG.

| Quantity | Value |
| --- | --- |
| Files | 159 |
| Windows asked | 512 |
| Positive events | 2,736 |
| Hits under the code rule | 1,422 |
| Misses | 1,314 |
| Hit rate | 1,422 / 2,736 = **51.97%** |
| Parsed events before merge | 4,098 |
| Merged reports | 3,418 |
| Reports with no positive overlap | 1,734 |
| Unmatched-report rate | 1,734 / 3,418 = **50.73%** |
| Of which overlap a negative label | 255 |
| Of which overlap nothing labeled | 1,479 |
| Same predictions, best same-channel report IoU > 0.7 | 692 / 2,736 = **25.29%** |

The code rule is coverage, not IoU. `utils/tuev_metrics.py` marks a ground-truth event as a hit when the union of same-channel overlaps covers at least 70% of that event's duration. Extra predicted time does not reduce the score. A long report over a short event can pass coverage and fail IoU. That coverage rule is the authors' own scorer in `acd2e6a`, not a local substitution. The paper's sentence says IoU. Partial overlaps (overlap > 0 but coverage < 0.7) are neither hits nor unmatched reports, so `1 - 50.73%` is not precision. Event precision in scorer v2 is the share of merged reports that overlap any positive event: 1,684/3,418 = 49.27%, and F1 with the coverage recall is 0.506.

Summed window time: model 11.35 h, tools 1.69 h, total 13.04 h. Mean rounds 2.83 (median 2). Round counts: 2→282, 3→144, 4→44, 5→12, 6→7, 7→8, 8→15. Ten windows ended on round 8 with `stopped_by_max_rounds` and therefore contributed no parsed events.

## By event class

Hits use the coverage rule. Filename prefixes are not classes. The eval README counts files that contain a class (SPSW 9, GPED 28, PLED 33, ARTF 46, EYEM 35, BCKG 89). Name prefixes in this run are bckg 67, spsw 40, pled 28, gped 24.

| Class | Code | Events | Hits | Recall |
| --- | --- | --- | --- | --- |
| SPSW | 1 | 216 | 112 | 51.85% |
| GPED | 2 | 1,633 | 936 | 57.32% |
| PLED | 3 | 887 | 374 | 42.16% |

Merged negative annotations, not used as the hit denominator: EYEM 289, ARTF 429, BCKG 2,844.

Ninety of 159 files have no positive event, including all 67 `bckg_*` files. They do not enter the recall denominator. They still produce 308 unmatched reports. Thirty-six of 67 background files emit at least one report. All 173 merged background reports are unmatched.

## Protocol limits

The agent is not scanned across the whole recording. `candidate_windows` merges every annotation, positive and negative, when the gap is at most 1 s, and the question names that interval. Times in the question are rounded to the nearest second. This is a time-cued detection task.

The question says "epileptic seizures". Classes 1–3 are epileptiform patterns. An electrographic seizure in the ACNS/Salzburg sense is a pattern on the order of 10 seconds, with a high discharge rate or evolution. A 1 s spike can be a true SPSW and still not be a seizure. The 1 s tools emit a class named `seiz`, and the prompt reports it at 0.50. The paper uses the same positive classes and the same seizure wording.

Every one of the 202 saved system prompts retrieved at least two chunks, all led by a duration definition: seizure pattern "usually >10 s" (164), a broken PDF fragment requiring about 10 seconds (18), electrographic status for about 10 minutes (17), or the Salzburg electrographic-seizure paragraph (3). Scores themselves are not stored. That text conflicts with the 1 s / 0.50 rule. See [models.md](models.md).

Channel names in this saved run were stored as parsed. `parse_events` now uppercases them and maps a reversed TCP pair onto the `CHANNEL_MAP` name. An unknown name is kept in `invalid_events` and is not scored. Re-parsing these logs with that rule moves the full-split coverage count from 1,422 to 1,424. The table above is the stored parse.

Final answers across 512 windows: 313 start with `(`, 141 are exactly `No events found`, 58 are neither. Three hundred ten windows have `messages: null`. That is a resume bug: `load_resumed_logs` used to set `messages` to null and the file writer then replaced the saved transcript. Ninety-one files lost every transcript; `bckg_037_a_` kept one of two. Tool-error counts from transcripts are a lower bound (12 windows with a tool error in the saved text, including `KeyError` on invented channel names). New runs append `messages/<stem>.messages.jsonl` before the file is scored, and resume reloads that file.

## What 51.97% can be compared with

Under a best same-channel report IoU > 0.7, this local run is 692/2,736 = 25.29%, and the authors' 36-file run is 146/447 = 32.66%. The paper's 69.30% does not match either figure, and it does not match the authors' own coverage rate of 73.83% on the 36 files they released. File-level bootstrap intervals are in [baseline.md](baseline.md).

The 51.97% folder is not the frozen baseline. It used the strict 0.50 prompt, nine tools, a requested `num_ctx` of 16384 that `/v1` ignored, and a continue-after-tools patch that only the `qwen3.8` name received. The run to keep is `tuev_authors_v2`, which has not been executed on the 159 files yet.

An oracle (`TUEV_oracle_run.py --rule both --threshold 0.5 --threshold 0.7`) calls `seizureNormalModel_OneSecond` or `seizureArtiBckgModel_OneSecond` on the same rounded windows and scores them with the same function. That run is not in `runs/` yet. Until it is, the gap between 58.17% and 73.83% on the paired 36 files cannot be split into detector error and planner error.
