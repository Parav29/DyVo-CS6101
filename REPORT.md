# DyVo reproduction on CODEC: full project report (CS6101)

Paper: *DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities* (Nguyen et al., EMNLP 2024), `cs6101_DyVo.pdf`.
Code: the authors' repository `thongnt99/DyVo` in `DyVo/`, edited only where needed (every edit is marked `# DYVO-REPRO`).
Status as of 1 Oct 2026: **all planned runs finished**. Midterm deadline: 3 Oct 2026.

Auto-generated detail (exact commands, timings, per-run health) lives in `repro_codec/REPORT.md`; this file is the narrative report.

---

## 1. Summary

We reproduced the CODEC rows of the paper's Tables 1 and 2 (L1 = 1e-5): the word-only baseline **LSR-w** and **DyVo** with REL, Mixtral and GPT-4 entity candidates. All four 100k-step rows land within **0.4 nDCG@10** of the paper, and three of four are slightly above it. Both headline claims hold in our runs:

1. Adding entities (DyVo) beats the word-only baseline.
2. LLM-generated entity candidates help most: GPT-4 > Mixtral > REL.

| CODEC, L1 = 1e-5, 100k steps | nDCG@10 ours / paper (diff) | nDCG@20 ours / paper (diff) | R@1000 ours / paper (diff) |
|---|---|---|---|
| LSR-w | 52.34 / 52.61 (-0.27) | 49.70 / 49.22 (+0.48) | 69.40 / 69.07 (+0.33) |
| DyVo (REL) | 53.70 / 53.40 (+0.30) | 50.78 / 51.15 (-0.37) | 70.82 / 70.60 (+0.22) |
| DyVo (Mixtral) | 55.13 / 54.90 (+0.23) | 52.68 / 52.82 (-0.14) | 73.45 / 73.20 (+0.25) |
| **DyVo (GPT-4), headline** | **56.84 / 56.46 (+0.38)** | 53.54 / 53.72 (-0.18) | 74.82 / 74.47 (+0.35) |

Test set: 42 CODEC topics, full 729,824-document corpus, top 10,000 retrieved per query, metrics from `ir_measures`.

Scope: **CODEC only.** Robust04 and TREC Core 2018 were skipped (licences), and no extensions were implemented. See sections 5 and 6.

---

## 2. What the midterm asked for, and where it is answered

The midterm structure (headline experiment, target, numbers, dataset, models, what existed, what was missing, then phases, assumptions, discrepancies) is answered as follows.

| Midterm item | Answer in this report |
|---|---|
| 1. Headline experiment | §3.1 |
| 2. Main reproduction target | §3.2 |
| 3. Target numbers | §3.3 |
| 4. Dataset | §3.4 |
| 5. Models and checkpoints | §3.5 |
| 6. What already existed | §3.6 |
| 7. What was missing and what we added | §3.7, §4.1 |
| Environment, dataset, model, training, evaluation phases | §4.2 to §4.6 |
| Comparison with the paper | §8 |
| Assumptions | §9 |
| Discrepancy analysis | §10 |
| Constraints, skipped work, problems and fixes | §5, §6, §7 |

---

## 3. Project definition

### 3.1 Headline experiment
DyVo extends a learned sparse retriever with Wikipedia **entity** dimensions. A DyVo head scores a small set of candidate entities against the encoder's hidden states, and those entity weights are appended to the 30,522 word-piece weights. Queries and documents are then matched on words and entities with a sparse dot product. The paper makes three claims:
- DyVo beats the word-only LSR baseline when both use linked (REL) entities (Table 1).
- Few-shot generative candidates (Mixtral, GPT-4) give the largest gains (Table 2).
- Both hold across three entity-rich datasets (we test one, see §5).

### 3.2 Reproduction target
Table 1 (L1 = 1e-5 rows) and Table 2, CODEC columns.

### 3.3 Target numbers (paper, CODEC)

| Row | nDCG@10 | nDCG@20 | R@1000 |
|---|---:|---:|---:|
| LSR-w, L1 = 1e-5 (Table 1) | 52.61 | 49.22 | 69.07 |
| DyVo (REL) | 53.40 | 51.15 | 70.60 |
| DyVo (Mixtral) | 54.90 | 52.82 | 73.20 |
| DyVo (GPT-4), best CODEC row with LaQue | 56.46 | 53.72 | 74.47 |

### 3.4 Dataset
- **Test:** CODEC, 42 topics, 729,824 documents, graded qrels (`irds:codec`).
- **Training:** the authors' released CODEC queries (5,681 with REL entities, plus Mixtral-generated synthetic queries) with MonoT5-3B teacher scores.

### 3.5 Models and checkpoints
All from the authors' Hugging Face dataset `lsr42/dyvo_data`:
- **Init:** DistilBERT query and document encoders pre-trained on MS MARCO (`dyvo_init/`).
- **Entity embeddings:** LaQue, 5,277,580 x 768 (16 GB).
- **Entities:** REL-linked entities for queries and documents, plus the authors' GPT-4 and Mixtral query-entity outputs.

No trained CODEC checkpoints were released, so every trained row is our own training run.

### 3.6 What already existed
The official code covers the model, entity head, collator, loss, trainer, evaluation and all experiment configs.

### 3.7 What was missing (see §4.1 for what we built)
Six bugs that stopped any run from starting, the LLM-candidate rows, budget configs for a shared GPU, and operational fixes for a shared server.

---

## 4. Everything done over the project

### 4.1 Build phase (before any real compute)
Built and checked on a low-resource laptop (CPU only, no downloads of weights or data), as a loop of small verified tasks tracked in `PROGRESS.md`, `NOTES.md` and `ERRORS.md`.
- **Offline verification gate** `scripts/verify.sh`: network blocked, 61 unit tests, a one-batch overfit test, and end-to-end smoke runs of the real training entry point with tiny random models. Result: `VERIFY PASSED`.
- **Six upstream bugs found and fixed** (details in `SANITY_REPORT.md`, `ERRORS.md`):
  1. `lsr/losses/__init__.py` imported three deleted modules, so no config could train from a fresh clone.
  2. `TransformerMLPeMLMConfig` never stored `entity_cls`, so the DyVo query encoder raised `AttributeError`.
  3. Eleven configs referenced a missing `dataset/trec_robust04_fold1`.
  4. The Wiki2Vec configs had `entity_dim: 768` for a 300-d table.
  5. Evaluation wrote to the authors' cluster path `/scratch-shared/$USER`; it now honours `LSR_SCRATCH_DIR`.
  6. `wandb.Settings(start_method=...)` failed on current wandb; pinned wandb 0.16.6.
- **Config checks:** 73 configs compose (43 official, 28 new for Tables 2 and 3, 2 smoke); 53 DyVo configs have matching entity widths.
- **Traceability:** `TRACEABILITY.md` maps the paper's sections and equations to code.
- **Big-PC scripts** for setup, data download, data checks, training, result collection and LLM entity generation, all dry-run tested.
- **Collapse probe** (paper App. A.2) at toy scale: the collapse did not reproduce, which is weak evidence only. We watched entity counts in the real runs instead.

### 4.2 Phase 1: environment
| Step | Output |
|---|---|
| Python 3.10 via `uv` (the machine only had 3.13) | interpreter |
| `scripts/big_pc/00_setup_env.sh` | `.venv-gpu`: torch 2.2.2+cu121, transformers 4.40.2, CUDA visible |
| `scripts/verify.sh` | `VERIFY PASSED` |

The lock file is installed with `--no-deps` because it pins protobuf versions pip rejects (ERRORS E5).

### 4.3 Phase 2: data
- Authors' data downloaded to `DyVo/dyvo_data/` (about 17 GB for CODEC plus LaQue).
- CODEC corpus from the public Hugging Face mirror `macavaney/codec` (sha256 verified): 729,824 documents, all 5,917 judged documents present.
- `02_check_data.py`: 0 errors (IDs aligned, teacher scores non-negative, entity table 5,277,580 x 768).
- LLM candidates merged with `lsr.entity.merge_candidates`: REL entities for the training queries, GPT-4 or Mixtral entities for the 42 test queries, with 0 falling back to REL.

### 4.4 Phase 3: model
- Official configs used unchanged apart from budget settings: LSR-w (`qmlp_dmlm_codec_...`) and DyVo (`qmlp_dmlm_emlm_laque_codec_..._entw_0.05`).
- MLP query encoder and MLM document encoder (DistilBERT), both initialised from `dyvo_init`.
- DyVo head with trainable lambda_ent initialised at 0.05, LaQue entity embeddings.
- **Collapse check:** the 100k DyVo run kept about 0.90 entities per query and 20.1 per document to the last step with finite loss (0.21). No collapse.

### 4.5 Phase 4: training
Settings kept from the paper: learning rate 5e-7, 10% warm-up with linear decay, KL distillation from MonoT5-3B, L1 = 1e-5 on documents only, 512-token documents, fp16, seed 42, effective batch 16.

| Run | Steps | Purpose | Wall clock |
|---|---|---|---|
| LSR-w `full100k` | 100k | reproduction baseline | 10.1 h on the last segment (resumed from step 60k) |
| DyVo `full100k` | 100k | the reproduction | 13.1 h on the last segment (resumed from step 20k) |
| LSR-w and DyVo `course20k` | 20k | reduced-budget fallback | done |
| LSR-w and DyVo `init` | 0 | how much CODEC training adds | done |

The GPT-4 and Mixtral rows are **inference only**: they use the REL-trained DyVo model and swap in LLM candidates for the test queries.

### 4.6 Phase 5: evaluation
Every query scores all 729,824 documents with the official rule (sum of query weight x document weight over matching ids); the top 10,000 are kept and `ir_measures` computes nDCG@10, nDCG@20 and R@1000.

The official evaluator took about 15 h per DyVo run. We wrote `scripts/big_pc/fast_eval.py`, which:
- uses the same model, collator, fp16 encoding and corpus order;
- encodes the corpus once and scores the REL, GPT-4 and Mixtral query sets together;
- takes about 1 h per model and resumes from partial scores.

Correctness checks:
1. Scores equal the official formula within 1e-6 on real DyVo encodings (`scripts/checks/check_scoring.py`).
2. Re-testing the LSR-w 20k model gives nDCG@10 52.19, nDCG@20 48.56 and R@1000 68.53, identical to the official `predict()` result.

### 4.7 Operational tooling built
- `supervise_train.sh`: restarts from the newest checkpoint, picks the GPU with the most free memory and the largest batch that fits (keeping effective batch 16), and runs the fast evaluator when training ends.
- `watch_events.sh`, `report_loop.sh`, `codec_report.py`, `codec_snapshot.sh`, `finalize_midterm.sh`: monitoring, report regeneration and snapshotting.
- Memory-mapped shared entity table (`lsr/models/entity_emb.py`).

---

## 5. Constraints we faced

**Data access**
- Robust04 and TREC Core 2018 need NIST/TREC licence agreements that we could not obtain in time. We did not use unofficial copies.
- No paid API and no local LLM. GPT-4 and Mixtral candidates are the authors' released outputs.
- No released trained checkpoints, so every trained row started from the MS MARCO init.

**Compute**
- Shared university GPU server with 2 x RTX A5000 (24 GB) instead of the paper's A100. Other users' jobs often left only 4 to 5 GB free per GPU, causing CUDA out-of-memory errors and forcing a per-device batch of 4 with 4 gradient-accumulation steps.
- Shared host RAM (125 GB). Each DyVo process originally loaded the 16 GB LaQue table twice (about 38 GB per process), so only one DyVo process fit at a time.

**Time**
- About five days of compute between environment setup (28 Sep) and the midterm (3 Oct).
- One 100k DyVo run takes 13 to 30 h depending on GPU sharing, and the official test took about 15 h per run.

**Scope**
- The midterm is reproduction only: no extensions or novel contributions.

---

## 6. What we skipped because of those constraints

| Skipped | Reason |
|---|---|
| Robust04 and TREC Core 2018 (a third of Tables 1 and 2, all their rows) | NIST-licensed corpora; the authors' entity and score files for them were downloaded but unused |
| L1 = 1e-3 and 1e-4 rows of Table 1 | each is an extra full training run; outside the L1 = 1e-5 headline |
| BM25, LaQue and Human candidate rows of Table 2 | configs and merged files exist but are not part of the CODEC headline; cheap inference-only add-ons if time remains |
| Table 3 (Token-Aggregation, DPR, JDS, Wiki2Vec, BLINK embeddings) | needs five more 6 to 22 GB embedding tables and a training run each; configs exist, never run |
| Regenerating entity candidates with an LLM | needs paid APIs or large local models; the authors' files were used (assumption A3) |
| Multiple seeds | one run per row at seed 42; the paper also reports single runs |
| Separate trained model per candidate source | GPT-4 and Mixtral rows reuse the REL-trained model (assumption A4); this also saved two 100k training runs |
| Mid-training best-checkpoint selection | none, same as the official configs |
| Extensions and novel contributions | midterm scope is reproduction |

---

## 7. Problems hit while running, and fixes

1. **Host out-of-memory kills (29 to 30 Sep).** The Linux OOM killer ended DataLoader workers, which killed DyVo 100k at step 24.7k, LSR-w 100k at step 64.4k and the DyVo 20k re-test. About one day of compute was lost. *Cause:* two private 16 GB copies of the entity table per process plus copy-on-write growth in 16 forked workers. *Fix:* the table is memory-mapped and shared; runs use 4 workers.
2. **Stale scratch directory crash at test time.** Cleanup of `queries.jsonl` and `docs/` shared one `try`, so a missing file skipped the `rmtree` and `mkdir` then raised `FileExistsError`. Fixed in `lsr/trainer/__init__.py` with independent cleanup and `exist_ok=True`.
3. **CUDA out-of-memory from other users' jobs.** The checkpoint-resuming supervisor restarts from the newest checkpoint on the freest GPU with the largest fitting batch.
4. **Slow official evaluation (about 15 h per DyVo run).** Replaced by the verified fast evaluator.
5. **DyVo init run failed** with CUDA OOM on the shared GPU; retried on a freer GPU and completed.

None of these changes the model, the loss or the scoring.

---

## 8. Final results and comparison with the paper

### 8.1 All rows

| Row | Steps | nDCG@10 ours / paper (diff) | nDCG@20 ours / paper (diff) | R@1000 ours / paper (diff) |
|---|---|---|---|---|
| LSR-w | 100k | 52.34 / 52.61 (-0.27) | 49.70 / 49.22 (+0.48) | 69.40 / 69.07 (+0.33) |
| DyVo (REL) | 100k | 53.70 / 53.40 (+0.30) | 50.78 / 51.15 (-0.37) | 70.82 / 70.60 (+0.22) |
| DyVo (Mixtral) | 100k | 55.13 / 54.90 (+0.23) | 52.68 / 52.82 (-0.14) | 73.45 / 73.20 (+0.25) |
| **DyVo (GPT-4)** | 100k | **56.84 / 56.46 (+0.38)** | 53.54 / 53.72 (-0.18) | 74.82 / 74.47 (+0.35) |
| LSR-w | 20k | 52.19 / 52.61 (-0.42) | 48.56 / 49.22 (-0.66) | 68.53 / 69.07 (-0.54) |
| DyVo (REL) | 20k | 52.71 / 53.40 (-0.69) | 49.56 / 51.15 (-1.59) | 70.25 / 70.60 (-0.35) |
| DyVo (Mixtral) | 20k | 54.71 / 54.90 (-0.19) | 52.71 / 52.82 (-0.11) | 73.31 / 73.20 (+0.11) |
| DyVo (GPT-4) | 20k | 56.12 / 56.46 (-0.34) | 53.31 / 53.72 (-0.41) | 74.65 / 74.47 (+0.18) |
| LSR-w, init | 0 | 49.44 | 44.89 | 62.64 |
| DyVo (REL), init | 0 | 51.01 | 46.43 | 64.75 |
| DyVo (Mixtral), init | 0 | 53.73 | 49.89 | 68.86 |
| DyVo (GPT-4), init | 0 | 55.31 | 50.99 | 70.43 |

The init rows have no paper counterpart. They show what CODEC training adds on top of the MS MARCO checkpoint.

### 8.2 What the numbers show
- **Claim 1 holds, entities help.** DyVo (REL) beats LSR-w at every budget: +1.57 nDCG@10 at init, +0.52 at 20k, and **+1.36 at 100k** (paper: +0.79).
- **Claim 2 holds, LLM candidates help most.** GPT-4 > Mixtral > REL at all three budgets. At 100k, GPT-4 adds +3.14 over REL (paper: +3.06) and Mixtral adds +1.43 (paper: +1.50).
- **Training budget matters, with diminishing returns.** 20k to 100k adds +0.4 to +1.0 nDCG@10 for DyVo; init to 20k adds +1.0 to +1.7.
- **Within noise.** Every 100k DyVo difference from the paper is at most 0.4 points. With 42 test queries and single-run reporting, about 1 point is within run-to-run noise.

### 8.3 Training health (100k runs)
- No NaN loss and no CUDA OOM inside the final segments.
- DyVo kept at least 0.88 entities per query and 19.5 per document throughout (no collapse).
- LSR-w has 0 entities, as expected.

---

## 9. Assumptions

| # | Assumption | Why | Expected effect |
|---|---|---|---|
| A1 | Gradient accumulation is equivalent to the paper's batch (4 per device x 4 = 16) | had to fit in about 6 GB beside other jobs | none: KL and L1 losses are batch means and there are no in-batch negatives |
| A2 | L1 warm-up counts micro-batches (T = 200,000 for 100k runs, 40,000 for 20k) | the regulariser steps once per micro-batch | same schedule as the paper |
| A3 | The authors' released data is the paper's data (REL linking, synthetic queries, MonoT5-3B scores, LaQue, GPT-4 and Mixtral candidates) | official release; regeneration needs paid APIs | none if the release matches the paper |
| A4 | GPT-4 and Mixtral rows reuse the REL-trained model and change only test-query candidates | the paper says LLM candidates are used on queries for inference; saves two 100k runs | differs only by run-to-run noise if the paper trained per row |
| A5 | Resuming from a checkpoint equals an uninterrupted run (DyVo from 20k, LSR-w from 60k) | both runs were killed by host OOM | weights, optimizer, scheduler and regulariser restored exactly; only post-resume data order differs |
| A6 | The last checkpoint is the final model | same as official configs | none |
| A7 | Test-time document batch size does not matter (DyVo 16, LSR-w 32, official 64) | GPU memory beside other users | none for LSR-w (identical 52.19); DyVo entity max-pooling over padding could shift scores very slightly |
| A8 | One run per row, seed 42 | the paper reports single runs; no time for seeds | about 1 nDCG@10 between seeds is plausible with 42 queries |
| A9 | The public Hugging Face CODEC copy is the official corpus | `ir_datasets` lists it as "on request" | same 729,824 documents; all judged documents present |
| A10 | Where paper and code disagree, the code is right (lambda_ent inside the ReLU, `normalize: "std"` is really L2, DyVo collator truncates queries at 512 tokens and LSR-w at 50) | the code produced the paper's numbers | none relative to the paper |
| A11 | Infrastructure changes do not change any computation (mmap entity table, fast evaluator) | needed to run at all on the shared server | verified: same tensor values, scores within 1e-6, LSR-w 20k re-test identical |

---

## 10. Discrepancy analysis

Our 100k DyVo rows are within 0.4 nDCG@10 of the paper, inside the noise expected from:
1. **Single run on 42 queries:** run-to-run variance of about 1 point.
2. **Numerical non-determinism:** fp16 and different GPU kernels (A5000 vs A100).
3. **Resumed training:** data order changed after the resume (A5).
4. **Test batch padding effect** on DyVo entity scores (A7).
5. **Library versions:** the paper states none; we used torch 2.2.2, transformers 4.40.2, ir_measures 0.4.3, wandb 0.16.6.

The 20k rows are 0.2 to 0.7 points below the paper on nDCG@10, as expected for one fifth of the training. The init rows are 2.4 to 3.2 points below, which shows CODEC training is doing real work and the pipeline is not trivially matching the paper.

---

## 11. Where everything is

| What | Where |
|---|---|
| This report | `REPORT.md` |
| Auto-generated results and exact commands per run | `repro_codec/REPORT.md` |
| Midterm write-up | `repro_codec/MIDTERM.md` |
| Per-run scores and rankings | `DyVo/outputs/<experiment>/fasteval/{test_run.trec,test_result.json}` |
| Training logs | `logs/<experiment>.log`, `logs/supervise_<experiment>.log` |
| Resolved configs and environment snapshot | `repro_codec/resolved_configs/`, `repro_codec/env/` |
| Scripts | `scripts/big_pc/` (`fast_eval.py`, `supervise_train.sh`, `codec_report.py`, ...) |
| Bugs, deviations, traceability, sanity checks | `ERRORS.md`, `NOTES.md`, `TRACEABILITY.md`, `SANITY_REPORT.md` |
| Handoff and context | `handoff.md`, `BIG_PC_HANDOFF.md`, `project_context.md` |
