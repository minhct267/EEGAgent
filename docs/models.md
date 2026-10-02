# Models

The saved local TUEV hit rate is 51.97% under the repository coverage scorer (1,422/2,736). On the 36 files the authors released, the same scorer gives qwen3.8 58.17% and qwen3-235b-a22b 73.83%. The paper's 69.30% is a different number and is not recovered from the released run. Details are in [tuev.md](tuev.md) and [baseline.md](baseline.md). The gap is not a clean ablation of parameter count: the saved local run also changed the question, the tool subset, the tool text, and the tool-result splice, and it ran at a context size it did not request. The current tools also fix a channel-labeling bug that the 235B run had (see [baseline.md](baseline.md)).

## What ran

From `.env`, and confirmed by the string `qwen3.8:27b` on every window in `runs/tuev_agent_ollama`:

| Setting | Value |
| --- | --- |
| `OLLAMA_PLANNER_BASE_URL` | `http://127.0.0.1:11434/v1` |
| `OLLAMA_PLANNER_MODEL` | `qwen3.8:27b` |
| `OLLAMA_REASONING_EFFORT` | `none` (client also sends `think: false`) |
| `OLLAMA_NUM_CTX` | 16384 requested in that run, through `/v1`, which ignores it. The server default applied instead (32,768 on the current Ollama 0.35.0). The baseline sends 65536 through `/api/chat`. |
| `OLLAMA_EMBED_MODEL` | `bge-m3:latest` on the same host. The released EEGAgent code also embedded with BGE-M3. The paper text names Qwen3-Embedding-8B. |

The logs do not record a quantization. The official Ollama tag `qwen3.8:27b` is a Q4_K_M build of a 27.3B model (`arch qwen35`, about 18 GB).

## The two planners

Qwen3-235B-A22B, the model cited by the paper (Yang et al., arXiv:2505.09388):

- Mixture-of-experts: 235B parameters total, 22B activated per token.
- 94 layers, 128 experts, 8 experts activated, no shared expert.
- Native context 32,768 tokens, extendable to 131,072 with YaRN.
- Thinking mode is on by default (`enable_thinking=True`). The authors' client called it non-streaming with the `enable_thinking: false` line commented out. DashScope rejects that request unless thinking is disabled, so the released 235B transcripts were generated with thinking off. Thinking off on the local run matches that reference. It is not, by itself, the reason the local hit rate is lower.

Qwen3.8-27B, the local tag (Hugging Face card `Qwen/Qwen3.8-27B`):

- Dense 27B model from a later generation than the paper. The paper is November 2025 and cites the original Qwen3 report.
- 64 layers, hidden size 5,120. Each of 16 groups is three Gated DeltaNet blocks and one Gated Attention block.
- Native context 262,144 tokens.
- Thinking is on by default. `reasoning_effort` defaults to `xhigh` (`xhigh`, `medium`, `low`). The card states that in multi-turn agent work a lower effort can shorten each turn and still increase failures and retries.

Per-token compute of the 235B model is on the order of its 22B active parameters, close to a 27B dense model. The capacity difference is the total parameter store and expert routing, not a 10× increase in FLOPs per token.

Public benchmark tables do not show Qwen3.8-27B as a weaker reasoner in general. Its own card reports GPQA Diamond 89.2, HLE 30.8, and LiveCodeBench v6 90.3. The later checkpoint Qwen3-235B-A22B-Thinking-2507, which is not the snapshot named in the paper, reports GPQA 81.1, HLE 18.2, and LiveCodeBench v6 74.1 on its card. Those figures are vendor-reported, on different harnesses, at different dates. They block the claim that "27B reasons worse than 235B" as a general fact. They do not explain this TUEV run.

## Why the saved local run is lower

Thinking was off on both sides of the 36-file comparison. The remaining differences that the logs and the code actually show are these.

Capacity and quantization. The local tag is Q4_K_M, about 18 GB. The paper does not name a quantization for the API model. Active parameters per token are similar (about 22B versus 27B). The 235B model still has a much larger total parameter store. On the paired files the coverage gap is 73.83% versus 58.17%, and the bootstrap intervals overlap (60.4–83.9 versus 45.5–68.9), so the point estimates are not a precise ranking.

The saved run asked for 16,384 tokens of context and did not get it. It called Ollama through `/v1`, which ignores `options` (`num_ctx`, `top_k`). The model ran at the server default instead, 32,768 tokens on the Ollama 0.35.0 now installed; the version used for the saved run is not logged. Above the loaded size, Ollama silently drops the oldest messages, including the first user turn with the question. Saved transcripts are 19,549 to 145,287 characters (median 29,901). The smoke run measured 2.2 to 2.6 characters per token, which puts 2 to 6 of the 202 saved windows over 32,768 tokens. The baseline uses `/api/chat` with `num_ctx` 65536 and `truncate: false`, so an over-long prompt is logged as `context_overflow` instead of being cut silently.

The tool dialect is the authors' splice of `<RETURN>` ... `<RESULT>` into the assistant message. Qwen3.8 often answered the next turn with nothing unless a user message told it to continue. The saved run already includes that patch, and the patch was gated on the model name `qwen3.8`. Details and counts are in [xml.md](xml.md). Ten windows still stopped on round 8 inside a tool call. The 235B run finished in two rounds on 108 of 120 windows. The baseline stops generation at `<RETURN>` and writes the result block as a user turn for every planner. `--harness authors_v1` is the ablation that puts the splice back.

The prompt was not the authors' prompt. The saved run added a `seiz >= 0.50` rule, exposed nine tools instead of all of them, and appended extra lines only when the model name matched `qwen3.8`. The baseline uses the authors' question and all tools. `--prompt strict` is the ablation that puts the 0.50 question back.

Retrieval still inserts clinical seizure definitions at a 10-second or 10-minute scale. All 202 saved prompts contain at least two such chunks (glossary "usually >10 s" in 164 of the first chunks). The positive classes are 1-second epileptiform labels. A model that follows the retrieved definition under-calls short SPSW. Similarity scores were not stored in that folder. The baseline stores the query, the chunk, and the score. BGE-M3 is the embedder the released code used, so switching embedders is not required to match the code. It remains a difference from the paper's Qwen3-Embedding-8B sentence. `--rag off` measures how much those chunks move the score.

## What is still missing

The frozen qwen3.8 run (`tuev_authors_v2`, seed 0, 159 files), the same 36 files at two more seeds, the four ablations, and the tool-only oracle. Commands are in [baseline.md](baseline.md). None of those is in `runs/` yet. A 235B rerun on the repaired harness is not available from this machine.
