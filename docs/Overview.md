# EEGAgent at a glance

EEGAgent is a planner-plus-toolbox system for EEG analysis. An LLM chooses tools, reads their scores, and writes a short answer (events, stages, or a report). Local models do the signal work. The LLM does not replace those models.

```mermaid
flowchart TB
  user[Question plus EDF] --> agent[EEGAgent planner]
  agent --> llm[Ollama planner LLM]
  llm -->|XML ReAct| tools[EEG toolbox]
  tools --> windows[Window and channel features]
  tools --> detectors[Local event models]
  agent --> rag[Local BGE-M3 plus FAISS]
  rag --> kb[Knowledge in RAG/docs]
  tools --> agent
  rag --> agent
  agent --> report[Tuples or short report]
  report --> score[TUEV event-level score]
```

## Who talks to whom

- **Planner** (`main.py` + `.env`): OpenAI-compatible chat to local Ollama or Ollama Cloud. It speaks the paper’s `<FUNCTION>` / `<ARGS>` loop, not native tool-calling.
- **Toolbox** (`tools/`): load/preprocess EDF, window features, and small PyTorch detectors (normal/abnormal, coarse 10 s, 1 s seizure, artifact type, sleep, MDD).
- **RAG** (`RAG/`): local `bge-m3` embeddings and a FAISS index over files in `RAG/docs`. This is the knowledge base, not the project `docs/` folder.
- **Eval**: `TUEV_eval.py` runs the agent on official TUEV pairs; `TUEV_oracle_run.py` runs tools only; `Sleep_eval.py` / `MDD_eval.py` cover the other public tasks.

```mermaid
flowchart LR
  subgraph inputs [Inputs]
    edf[EDF recordings]
    rec[TUEV .rec labels]
    kb[RAG/docs]
  end
  subgraph core [Runtime]
    planner[Planner LLM]
    toolbox[Registered tools]
    index[FAISS index]
  end
  subgraph outputs [Outputs]
    answer[Final text]
    runs[runs/ logs and scores]
  end
  edf --> toolbox
  kb --> index
  planner --> toolbox
  toolbox --> planner
  index --> planner
  rec --> runs
  answer --> runs
```

## Where data lives

| Location | Role |
| --- | --- |
| `D:\Datasets\TUH-EEG\TUEV` and `TUAB` | External datasets. Read-only. |
| `data/` | Small in-repo samples for smoke tests. |
| `tools/localModels/*.pth` | Detector weights. |
| `RAG/docs`, `RAG/faiss.index` | Knowledge the agent can retrieve. |
| `runs/` | Generated eval logs. Gitignored. Recreate with the eval scripts. |
| `docs/` | Human runbook and this map. Not used at runtime. |

## Three ways to check the system

```mermaid
flowchart TD
  smoke[Smoke: env, planner, embed, tools] --> ready[Ready to eval]
  agentEval[Agent eval on TUEV windows] --> hit[Hit rate vs .rec]
  oracle[Oracle: tools only, no LLM] --> ceiling[Tool ceiling under the same score]
```

1. **Smoke** — is Ollama up, is `bge-m3` answering, do tools load? See [Note.md](Note.md).
2. **Agent eval** — the planner must pick tools and emit `(channel, start, end)` tuples. Scored against TUEV labels with overlap 0.7 after merging nearby reports.
3. **Oracle** — same score, but the script calls the seizure tools itself. Use this to separate “the detector missed” from “the LLM did not call the right tool”.

## Design that stays fixed

The live path is XML ReAct, a discharge-tool subset for TUEV, and gated extras for MiniMax (tool results as a user turn) and Qwen3.8 (continue-after-tools). Those are runtime contracts, not leftovers from a one-off experiment.

How to run and switch models: [Note.md](Note.md).
