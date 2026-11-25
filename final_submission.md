# Final submission: DyVo reproduction on CODEC (CS6101)

Paper: *DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities* (Nguyen et al., EMNLP 2024), `cs6101_DyVo.pdf`.
Authors' code: `thongnt99/DyVo` (in `DyVo/`), edited only where needed; every edit is marked `# DYVO-REPRO`.
Status as of 3 Oct 2026: all planned runs finished. Scope is **CODEC only**, reproduction only (no extensions).

Companion files: `REPORT.md` (narrative), `repro_codec/REPORT.md` (auto-generated commands, timings, per-run health), `repro_codec/MIDTERM.md`, `ERRORS.md`, `NOTES.md`, `TRACEABILITY.md`, `SANITY_REPORT.md`.

---

## 1. Summary

We reproduced the CODEC rows of the paper's Tables 1 and 2 (L1 = 1e-5): the word-only baseline **LSR-w**, and **DyVo** with REL, Mixtral and GPT-4 entity candidates. All four 100k-step rows are within **0.4 nDCG@10** of the paper; three of four are slightly above it. Both headline claims hold:

1. Adding entities (DyVo) beats the word-only baseline.
2. LLM-generated entity candidates help most: GPT-4 > Mixtral > REL.

Skipped: Robust04 and TREC Core 2018 (licences), L1 = 1e-3/1e-4 rows, BM25/LaQue/Human candidate rows, Table 3 (other entity embeddings), LLM candidate regeneration, multiple seeds. See §9.

---

## 2. What the method is

DyVo extends a learned sparse retriever with Wikipedia **entity** dimensions. A DyVo head scores a small set of candidate entities against the encoder's hidden states; those entity weights are appended to the 30,522 word-piece weights. Queries and documents are matched on words and entities with a sparse dot product. The paper claims: (a) DyVo beats word-only LSR with linked (REL) entities, (b) few-shot generative candidates (Mixtral, GPT-4) give the largest gains, (c) both hold across three entity-rich datasets (we test one).

Reproduction target: Table 1 (L1 = 1e-5 rows) and Table 2, CODEC columns.

---

## 3. What came from the authors, and what is ours

### 3.1 Authors' material used
| Item | Source | Notes |
|---|---|---|
| Code (model, entity head, collator, loss, trainer, evaluation, configs) | `thongnt99/DyVo` | 43 official configs; used unchanged apart from budget settings |
| Init checkpoints | HF dataset `lsr42/dyvo_data`, `dyvo_init/` | DistilBERT query/document encoders pre-trained on MS MARCO |
| Entity embeddings | same, LaQue | 5,277,580 x 768, 16 GB |
| REL-linked entities | same | for queries and documents |
| Synthetic training queries | same | Mixtral-generated, plus 5,681 CODEC queries with REL entities |
| Teacher scores | same | MonoT5-3B |
| GPT-4 and Mixtral query-entity candidates | same | authors' released outputs; we did not regenerate |
| Hyperparameters | paper and official configs | see §6 |

No trained CODEC checkpoints were released, so every trained row is our own training run.

### 3.2 Our material
- Fixes to six upstream bugs (§5.1).
- 28 new configs for Tables 2 and 3 plus 2 smoke configs (73 configs compose in total); budget configs for a shared GPU.
- Offline verification gate `scripts/verify.sh`, 61 unit tests, overfit and smoke tests.
- `scripts/big_pc/` scripts: setup, data download/check, training, result collection, LLM-candidate merge.
- `fast_eval.py` (fast evaluator), `supervise_train.sh` (resuming supervisor), monitoring and report scripts.
- Memory-mapped shared entity table (`lsr/models/entity_emb.py`).
- Stale-scratch fix in `lsr/trainer/__init__.py`.
- `TRACEABILITY.md` mapping paper sections/equations to code.

---

## 4. How we got the datasets

- **Authors' data:** downloaded from HF `lsr42/dyvo_data` to `DyVo/dyvo_data/` (about 17 GB for CODEC plus LaQue).
- **CODEC corpus:** the public HF mirror `macavaney/codec`, sha256 verified. `ir_datasets` lists CODEC as "on request" (assumption A9). 729,824 documents; all 5,917 judged documents present.
- **Test set:** CODEC, 42 topics, graded qrels (`irds:codec`).
- **Checks:** `02_check_data.py` reported 0 errors (IDs aligned, teacher scores non-negative, entity table 5,277,580 x 768).
- **LLM candidates:** merged with `lsr.entity.merge_candidates`: REL entities for training queries, GPT-4 or Mixtral entities for the 42 test queries; 0 test queries fell back to REL.
- **Not obtained:** Robust04 and TREC Core 2018 need NIST/TREC licence agreements we could not get in time; we did not use unofficial copies. (Authors' entity and score files for them were downloaded but unused.)

---

## 5. What we did, in order

### 5.1 Build phase (laptop, CPU only, no weights or data)
Small verified tasks tracked in `PROGRESS.md`, `NOTES.md`, `ERRORS.md`. Result of the gate: `VERIFY PASSED` (network blocked, 61 unit tests, one-batch overfit, end-to-end smoke runs with tiny random models).

Six upstream bugs found and fixed:
1. `lsr/losses/__init__.py` imported three deleted modules, so no config trained from a fresh clone.
2. `TransformerMLPeMLMConfig` never stored `entity_cls` (`AttributeError` in the DyVo query encoder).
3. Eleven configs referenced a missing `dataset/trec_robust04_fold1`.
4. Wiki2Vec configs had `entity_dim: 768` for a 300-d table.
5. Evaluation wrote to the authors' cluster path `/scratch-shared/$USER`; now honours `LSR_SCRATCH_DIR`.
6. `wandb.Settings(start_method=...)` failed on current wandb; pinned wandb 0.16.6.

Config checks: 73 configs compose; 53 DyVo configs have matching entity widths. A toy-scale collapse probe (paper App. A.2) did not reproduce the collapse (weak evidence only), so we watched entity counts in real runs.

### 5.2 Phase 1: environment
Python 3.10 via `uv` (machine had only 3.13); `scripts/big_pc/00_setup_env.sh` gave `.venv-gpu` with torch 2.2.2+cu121, transformers 4.40.2, CUDA visible. The lock file is installed with `--no-deps` because it pins protobuf versions pip rejects (ERRORS E5). Verify gate passed again.

### 5.3 Phase 2: data
See §4.

### 5.4 Phase 3: model
- LSR-w (`qmlp_dmlm_codec_...`) and DyVo (`qmlp_dmlm_emlm_laque_codec_..._entw_0.05`) official configs.
- MLP query encoder, MLM document encoder (DistilBERT), both initialised from `dyvo_init`.
- DyVo head with trainable lambda_ent initialised at 0.05, LaQue entity embeddings.

### 5.5 Phase 4: training
| Run | Steps | Purpose | Wall clock |
|---|---|---|---|
| LSR-w `full100k` | 100k | reproduction baseline | 10.1 h last segment (resumed from 60k) |
| DyVo `full100k` | 100k | the reproduction | 13.1 h last segment (resumed from 20k) |
| LSR-w, DyVo `course20k` | 20k | reduced-budget fallback | done |
| LSR-w, DyVo `init` | 0 | what CODEC training adds | done |

GPT-4 and Mixtral rows are **inference only**: they use the REL-trained DyVo model with LLM candidates swapped in for test queries. Total compute across killed attempts and resumes was not tallied (about one day lost to OOM kills, §7).

### 5.6 Phase 5: evaluation
Every query scores all 729,824 documents with the official rule (sum of query weight x document weight over matching ids); top 10,000 kept; `ir_measures` gives nDCG@10, nDCG@20, R@1000.

The official evaluator took about 15 h per DyVo run. `scripts/big_pc/fast_eval.py` uses the same model, collator, fp16 encoding and corpus order, encodes the corpus once, scores REL/GPT-4/Mixtral query sets together, and takes about 1 h per model with resume. Correctness: scores equal the official formula within 1e-6 on real DyVo encodings (`scripts/checks/check_scoring.py`); re-testing LSR-w 20k gives nDCG@10 52.19, nDCG@20 48.56, R@1000 68.53, identical to the official `predict()`.

### 5.7 Operational tooling
`supervise_train.sh` (restart from newest checkpoint, pick freest GPU and largest fitting batch while keeping effective batch 16, run fast eval at the end), `watch_events.sh`, `report_loop.sh`, `codec_report.py`, `codec_snapshot.sh`, `finalize_midterm.sh`.

---

## 6. Settings

Kept from the paper: learning rate 5e-7, 10% warm-up with linear decay, KL distillation from MonoT5-3B, L1 = 1e-5 on documents only, 512-token documents, fp16, seed 42, effective batch 16.
Changed for the shared GPU: per-device batch 4 with 4 gradient-accumulation steps, 4 DataLoader workers, test document batch 16 (DyVo) / 32 (LSR-w) vs official 64.

---

## 7. Constraints and problems

**Constraints.** Shared university server with 2 x RTX A5000 (24 GB) instead of an A100; other users often left only 4-5 GB free per GPU. Shared 125 GB host RAM (each DyVo process originally loaded the 16 GB LaQue table twice, about 38 GB). About five days of compute (28 Sep to 3 Oct). No paid API or local LLM. No released trained checkpoints.

**Problems and fixes.**
1. *Host OOM kills (29-30 Sep).* The Linux OOM killer ended DataLoader workers, killing DyVo 100k at step 24.7k, LSR-w 100k at step 64.4k, and the DyVo 20k re-test; about one day lost. Cause: two private 16 GB table copies per process plus copy-on-write growth in 16 forked workers. Fix: memory-mapped shared table, 4 workers.
2. *Stale scratch crash at test time.* Cleanup of `queries.jsonl` and `docs/` shared one `try`, so a missing file skipped `rmtree` and `mkdir` raised `FileExistsError`. Fixed with independent cleanup and `exist_ok=True`.
3. *CUDA OOM from other users' jobs.* Handled by the resuming supervisor.
4. *Slow official evaluation (~15 h per DyVo run).* Replaced by the verified fast evaluator.
5. *DyVo init run failed with CUDA OOM;* retried on a freer GPU and completed.

None of these changes the model, loss or scoring.

---

## 8. Results and comparison with the paper

Test set: 42 CODEC topics, full 729,824-document corpus, top 10,000 per query.

### 8.1 Paper targets (CODEC)
| Row | nDCG@10 | nDCG@20 | R@1000 |
|---|---:|---:|---:|
| LSR-w, L1 = 1e-5 (Table 1) | 52.61 | 49.22 | 69.07 |
| DyVo (REL) | 53.40 | 51.15 | 70.60 |
| DyVo (Mixtral) | 54.90 | 52.82 | 73.20 |
| DyVo (GPT-4) | 56.46 | 53.72 | 74.47 |

### 8.2 Ours vs paper
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

Init rows have no paper counterpart; they show what CODEC training adds on top of the MS MARCO checkpoint.

### 8.3 What the numbers show
- **Claim 1 (entities help) holds.** DyVo (REL) minus LSR-w, nDCG@10: +1.57 at init, +0.52 at 20k, **+1.36 at 100k** (paper: +0.79).
- **Claim 2 (LLM candidates help most) holds.** GPT-4 > Mixtral > REL at every budget. At 100k, GPT-4 adds +3.14 over REL (paper +3.06) and Mixtral adds +1.43 (paper +1.50).
- **Training budget has diminishing returns.** 20k to 100k adds +0.4 to +1.0 nDCG@10 for DyVo (Mixtral +0.42, GPT-4 +0.72, REL +0.99). Init to 20k adds +0.8 to +1.7 for DyVo (GPT-4 +0.81, Mixtral +0.98, REL +1.70) and +2.75 for LSR-w.
- **Within noise.** Every 100k DyVo difference from the paper is at most 0.4 points. With 42 queries and single-run reporting, about 1 point is within run-to-run noise.
- **20k rows** are 0.2 to 0.7 below the paper on nDCG@10, as expected for one fifth of the training. **Init rows** are 1.2 to 3.2 below the paper's trained numbers (LSR-w -3.17, REL -2.39, Mixtral -1.17, GPT-4 -1.15), so CODEC training does real work and the pipeline does not trivially match the paper.

### 8.4 Training health (100k runs)
No NaN loss and no CUDA OOM inside the final segments. DyVo kept at least 0.88 entities per query and 19.5 per document throughout (final: about 0.90 and 20.1, loss 0.21), so no collapse. LSR-w has 0 entities, as expected.

### 8.5 Sources of discrepancy
1. Single run on 42 queries (about 1 point of variance).
2. fp16 and different GPU kernels (A5000 vs A100).
3. Resumed training changed data order after the resume (A5).
4. Test batch padding effect on DyVo entity max-pooling (A7).
5. Library versions are unstated in the paper; we used torch 2.2.2, transformers 4.40.2, ir_measures 0.4.3, wandb 0.16.6.

---

## 9. What we skipped, and why

| Skipped | Reason |
|---|---|
| Robust04 and TREC Core 2018 (a third of Tables 1 and 2) | NIST-licensed corpora not obtained |
| L1 = 1e-3 and 1e-4 rows of Table 1 | each needs an extra full training run; outside the L1 = 1e-5 headline |
| BM25, LaQue and Human candidate rows of Table 2 | not part of the CODEC headline; configs and merged files exist; cheap inference-only add-ons |
| Table 3 (Token-Aggregation, DPR, JDS, Wiki2Vec, BLINK embeddings) | needs five more 6-22 GB tables and a training run each; configs exist, never run |
| Regenerating entity candidates with an LLM | needs paid APIs or large local models; authors' files used (A3) |
| Multiple seeds | one run per row at seed 42; the paper also reports single runs |
| Separate trained model per candidate source | GPT-4/Mixtral rows reuse the REL-trained model (A4); saved two 100k runs |
| Best-checkpoint selection | none, same as official configs |
| Extensions and novel contributions | midterm scope is reproduction |

---

## 10. Assumptions

| # | Assumption | Why | Expected effect |
|---|---|---|---|
| A1 | Gradient accumulation (4 x 4 = 16) equals the paper's batch | had to fit in about 6 GB beside other jobs | none: KL and L1 losses are batch means, no in-batch negatives |
| A2 | L1 warm-up counts micro-batches (T = 200,000 for 100k runs, 40,000 for 20k) | the regulariser steps once per micro-batch | same schedule as the paper |
| A3 | Authors' released data is the paper's data (REL linking, synthetic queries, MonoT5-3B scores, LaQue, GPT-4/Mixtral candidates) | official release; regeneration needs paid APIs | none if the release matches the paper |
| A4 | GPT-4/Mixtral rows reuse the REL-trained model, changing only test-query candidates | paper says LLM candidates are used on queries at inference; saves two 100k runs | run-to-run noise only if the paper trained per row |
| A5 | Resuming from a checkpoint equals an uninterrupted run (DyVo from 20k, LSR-w from 60k) | both killed by host OOM | weights, optimizer, scheduler, regulariser restored exactly; only post-resume data order differs |
| A6 | Last checkpoint is the final model | same as official configs | none |
| A7 | Test document batch size does not matter (DyVo 16, LSR-w 32, official 64) | GPU memory beside other users | none for LSR-w (identical 52.19); DyVo entity max-pooling over padding could shift scores very slightly |
| A8 | One run per row, seed 42 | paper reports single runs; no time for seeds | about 1 nDCG@10 between seeds is plausible with 42 queries |
| A9 | Public HF CODEC copy is the official corpus | `ir_datasets` lists it as "on request" | same 729,824 documents; all judged documents present |
| A10 | Where paper and code disagree, the code is right (lambda_ent inside the ReLU; `normalize: "std"` is really L2; DyVo collator truncates queries at 512 tokens, LSR-w at 50) | the code produced the paper's numbers | none relative to the paper |
| A11 | Infrastructure changes do not change any computation (mmap table, fast evaluator) | needed to run on the shared server | verified: same tensor values, scores within 1e-6, LSR-w 20k re-test identical |

---

## 11. Where everything is

| What | Where |
|---|---|
| This file | `final_submission.md` |
| Narrative report | `REPORT.md` |
| Auto-generated results and exact commands | `repro_codec/REPORT.md` |
| Midterm write-up | `repro_codec/MIDTERM.md` |
| Per-run scores and rankings | `DyVo/outputs/<experiment>/fasteval/{test_run.trec,test_result.json}` |
| Training logs | `logs/<experiment>.log`, `logs/supervise_<experiment>.log` |
| Resolved configs and environment snapshot | `repro_codec/resolved_configs/`, `repro_codec/env/` |
| Scripts | `scripts/big_pc/` |
| Bugs, deviations, traceability, sanity checks | `ERRORS.md`, `NOTES.md`, `TRACEABILITY.md`, `SANITY_REPORT.md` |
| Handoff and context | `handoff.md`, `BIG_PC_HANDOFF.md`, `project_context.md` |
