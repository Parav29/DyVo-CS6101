# HANDOFF: DyVo reproduction (CS6101), moving to a new computer

_Written 2026-09-29 ~16:00 IST from the university GPU server (`24b4515@…`, project dir `~/Dyvo_cs6101`). Read this file top to bottom before running anything. Where it disagrees with the older docs (`BIG_PC_HANDOFF.md`, `project_context.md`, `repro_codec/REPORT.md`), this file is the newer one._

**Midterm deadline: 3 Oct 2026.** That date shapes the priorities in §6.

---

## 0. TL;DR

- **Goal:** reproduce the CODEC rows of the DyVo paper (`cs6101_DyVo.pdf`): LSR-w baseline vs DyVo with REL / GPT-4 / Mixtral entity candidates, L1 = 1e-5. Target nDCG@10: 52.61 / 53.40 / 56.46 / 54.90.
- **Code is done and verified.** Everything left is *compute*: training and testing on a GPU.
- **Only 3 things exist as real results so far** (all from the old server, see §3): LSR-w 20k-step (52.19), LSR-w init (49.44), and nothing yet for DyVo. The two paper-length (100k-step) runs were ~49 % and ~22 % done when this was written.
- **Git has the code but not the data or model outputs.** The new machine must re-download data (§4) and either re-train or copy checkpoints across (§5).
- **We skip** Robust04, TREC Core 2018 (NIST licences), the L1 sweep, Table 3 (other embedding tables), BM25/LaQue/Human candidate rows, and any proposed extension. Full list in §2.
- **No paid API or local LLM is needed.** All GPT-4 / Mixtral entity outputs ship in the authors' data repo.

---

## 1. Project background (what a fresh reader needs)

DyVo = "Dynamic Vocabularies for Learned Sparse Retrieval with Entities". A learned sparse retriever (LSR-w: DistilBERT MLM head over 30,522 word-pieces) is extended with **entity dimensions**. For each query/doc a candidate set of Wikipedia entities is scored against the encoder's hidden states and the entity weights are appended to the word-piece weights.

- Upstream code: `DyVo/` (github.com/thongnt99/DyVo, with our fixes; every change is marked `# DYVO-REPRO`).
- Our repo: `origin = https://github.com/Parav29/DyVo-CS6101`.
- Framework: Hydra configs + HF Trainer. Entry point: `cd DyVo && python -m lsr.train +experiment=<name> …`.
- Teacher: MonoT5-3B scores (shipped). Training queries: 5.6k CODEC queries with REL entities + 10k Mixtral synthetic queries. Test: 42 CODEC topics, full 729,824-doc corpus, top-10,000.
- **Training always uses REL entities.** The GPT-4 / Mixtral rows differ only in the *test-time query candidates*, so they are **inference-only runs on an already trained DyVo model**. No retraining per row.

### Map of the docs already in the repo
| File | Content |
|---|---|
| `BIG_PC_HANDOFF.md` | Original run guide (written before anything was trained; partly outdated: assumes batch 16 / big GPU) |
| `project_context.md` | Snapshot from 2026-09-29 14:25 (hardware, experiments table, bugs) |
| `repro_codec/REPORT.md` | Auto-generated results + exact commands per run (regenerate with `codec_report.py`) |
| `repro_codec/MIDTERM.md` | Midterm write-up skeleton: phases, commands, expected outputs |
| `NOTES.md` | Deviations from upstream, code↔paper map (24 KB) |
| `ERRORS.md` | E1–E7 error log with fixes (+ uncommitted additions) |
| `SANITY_REPORT.md`, `TRACEABILITY.md` | Unit/behaviour checks; paper equations → code |
| `DYVO_IMPLEMENTATION_PROMPT.md` | The original build prompt |

---

## 2. Scope: what we do and what we skip

### In scope (CODEC only, L1 = 1e-5)
| # | Row | Experiment suffix (`E` = `qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05`; `W` = `qmlp_dmlm_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001`) | Needs training? | Paper nDCG@10 / @20 / R@1k |
|---|---|---|---|---|
| 1 | LSR-w, paper-length | `W_full100k` | yes, 100k steps | 52.61 / 49.22 / 69.07 |
| 2 | DyVo (REL), paper-length | `E_full100k` | yes, 100k steps | 53.40 / 51.15 / 70.60 |
| 3 | **DyVo (GPT-4 candidates), headline** | `E_full100k_eval_gpt4` | no, inference on row 2 | 56.46 / 53.72 / 74.47 |
| 4 | DyVo (Mixtral candidates) | `E_full100k_eval_mistral` | no, inference on row 2 | 54.90 / 52.82 / 73.20 |
| 5 | LSR-w, 20k-step "course budget" | `W_course20k` | yes, **done** | (same paper row) |
| 6 | DyVo (REL), 20k | `E_course20k` (+ `_eval_rel` re-test) | yes, trained; **test still missing** | (same) |
| 7 | DyVo GPT-4 / Mixtral on the 20k model | `E_course20k_eval_gpt4`, `_eval_mistral` | no, inference | (same) |
| 8 | LSR-w init / DyVo init (authors' MS MARCO checkpoint, no CODEC training) | `W_init`, `E_init` | no | shows what CODEC training adds |

Rows 5–8 are the **fallback / context rows**. If the new machine is too slow for rows 1–4 by the deadline, rows 5–7 already give a complete "LSR-w vs DyVo(REL/GPT-4/Mixtral)" story at a reduced budget (see §6).

### Skipped (and why; write these into the report as scope limits)
| Skipped | Reason |
|---|---|
| **Robust04** and **TREC Core 2018** (all their rows in Tables 1–3) | Need NIST-licensed corpora (TREC Disks 4&5, Washington Post v2). We do not use unofficial copies. The authors' entity/score files for them were downloaded but are unused. |
| **L1 = 1e-3 and 1e-4** rows (rest of Table 1) | Outside the L1 = 1e-5 headline; each is an extra full training run. |
| **Table 2 BM25 / LaQue / Human candidate rows** | Configs and merged candidate files exist (`*_bm25`, `*_laquecand`, `*_human`) but they are not part of the CODEC headline. They are cheap add-ons (inference only) if time remains. |
| **Table 3** (Token-Aggregation, DPR, JDS, Wiki2Vec, BLINK embeddings) | Need 5 more entity-embedding tables (6.3–21.6 GB each) and a training run each. Configs exist, never run. |
| **Regenerating entity candidates with an LLM** (`generate_entities_llm.py`, `05_generate_entities.sh`) | Not needed; authors' files used. Doing it would be a deviation. |
| **Extensions / novel contributions** | Midterm is reproduction only. |
| **Multiple seeds** | Paper reports one run; we use seed 42, one run. |

### Deliberate deviations from the paper (already recorded in `repro_codec/REPORT.md`)
1. Batch = 4 per device × 4 gradient-accumulation = **16 effective** (paper batch 16), because the old GPUs were shared. **On a dedicated big GPU you may use `per_device_train_batch_size=16`, `gradient_accumulation_steps=1`** (same effective batch, faster). Note the `full100k` YAML sets `gradient_accumulation_steps: 4`, so if you change the per-device batch you must also override that (§5.3).
2. L1 warm-up `T = 200000` in the `full100k` configs (regulariser steps once per micro-batch, so 50k optimizer steps × 4). **If you switch to accumulation 1, set `loss.d_regularizer.T=50000` (or the value that matches your micro-batch count) to keep the same schedule.**
3. No mid-training evaluation (it only logged, no best-checkpoint selection).
4. fp16, WANDB offline.
5. Course-budget 20k-step runs (10 % warm-up, linear decay scaled to 20k).
6. Eval batch 8 instead of 64 on the old server (scores don't depend on it). **Use the default 64 on the new machine.**

---

## 3. State at handoff (2026-09-29 ~16:00 IST)

### 3.1 Finished results (all on CODEC test, 42 topics)
| Run | nDCG@10 | nDCG@20 | R@1000 | vs paper nDCG@10 |
|---|---|---|---|---|
| LSR-w, 20k steps | **52.19** | 48.56 | 68.53 | 52.61 (−0.42) |
| LSR-w, init (no training) | 49.44 | 44.89 | 62.64 | (−3.17) |

Both lie inside the "1–2 points" tolerance, so the pipeline (data, teacher scores, loss, eval) is validated for LSR-w.
Files: `DyVo/outputs/<exp>/09-29-2026/test_result.json` (**git-ignored, not in the repo**). The numbers above are copied here so they are not lost.

### 3.2 Trained but *not yet tested*
- **DyVo (REL) 20k**: training finished at 07:42 (query entities ≈ 0.92, doc entities ≈ 20, loss 0.228: no entity collapse, no NaN). Its built-in test **crashed** with `FileExistsError` (stale `tmp_eval/docs` dir; fixed in `DyVo/lsr/trainer/__init__.py`, see §7 bug 1). Model weights are at `DyVo/outputs/E_course20k/09-29-2026/model/` (510 MB).
- A re-test of that model (`E_course20k_eval_rel`) was running on the old server at only **~1 s/it, ETA ~24 h** (eval batch 8 on a shared GPU, 6 % done). It will be killed/abandoned; on the new machine at batch 64 the LSR-w test took **48 min**, so expect ≈ 1 h.

### 3.3 Still training on the old server (will not survive the move)
| Run | Progress | Speed on the shared server | Remaining there |
|---|---|---|---|
| LSR-w 100k (`W_full100k`) | step ~48.8k / 100k | ~4.0 s/it | ~57 h |
| DyVo(REL) 100k (`E_full100k`) | step ~22.5k / 100k | ~3.7 s/it | ~79 h |

Latest checkpoints (≈1.5 GB each, contain `optimizer.pt`, `scheduler.pt`, `trainer_state.json`): `W_full100k/09-29-2026/checkpoint-45000`, `E_full100k/09-29-2026/checkpoint-20000`. Newer ones appear every 5k steps (`save_total_limit: 2`). The old server's orchestrators (`run_codec_phase2.sh`, `run_codec_fix.sh`, tmux sessions `train`, `dyvo`) keep running until someone stops them. **If the old server keeps running while you work on the new machine, do not let both write the same files; they only share NFS `~`, so the new machine's copy of the repo is separate.**

### 3.4 Never succeeded
- `E_init` (DyVo with the authors' init weights): CUDA OOM on the shared GPU (other users). Not a code bug; retry on a free GPU.
- GPT-4/Mixtral evals of the 20k and 100k DyVo models: not started (need the trained DyVo model first).

---

## 4. Step-by-step setup on the new computer

### 4.0 Minimum hardware
| Resource | Need | Note |
|---|---|---|
| GPU | 1 × ≥ 16 GB (24 GB comfortable; A100 ideal). With 6 GB you can only run batch 4 × 4 accum. | fp16 training peak ≈ 6 GB at micro-batch 4; more at 16 |
| **CPU RAM** | **≥ 48 GB free for any DyVo run** (the 16 GB LaQue table is `torch.load`ed whole; process ≈ 38 GB). LSR-w alone needs far less. | Only **one DyVo process at a time** below ~90 GB |
| Disk | ~45 GB: `dyvo_data` 17 GB (if only CODEC + LaQue) + CODEC corpus 5.2 GB + venv 5.6 GB + checkpoints (each 1.5 GB, ×2 kept per run) + scratch for eval (~15 GB) | |
| Network | HF Hub access; ~23 GB of downloads | |

### 4.1 Get the code (from git)
```bash
git clone https://github.com/Parav29/DyVo-CS6101.git Dyvo_cs6101 && cd Dyvo_cs6101
```
> ⚠️ **Before you leave the old server, commit and push. A fresh clone will NOT contain the files below** (verified with `git status` on 2026-09-29; the last commit is only `158b65b iter 11`, the build-machine work).
> - **Modified (8):** `DyVo/lsr/trainer/__init__.py` (**the stale-scratch bug fix, critical**), `scripts/big_pc/{00_setup_env.sh,02_check_data.py,03_train.sh,run_priority.sh}`, `scripts/verify.sh`, `ERRORS.md`, `BIG_PC_HANDOFF.md`.
> - **Untracked but required:** all 11 experiment YAMLs `DyVo/lsr/configs/experiment/*_course20k*.yaml`, `*_full100k*.yaml`, `*_init.yaml` (incl. `_eval_rel/_eval_gpt4/_eval_mistral`); `scripts/big_pc/{run_codec_course.sh,run_codec_phase2.sh,run_codec_fix.sh,wait_and_train.sh,watch_events.sh,codec_report.py,codec_snapshot.sh}`; `repro_codec/` (REPORT, MIDTERM, resolved configs, env snapshot); `project_context.md`; `handoff.md`.
> - **Do not commit** (already git-ignored or huge): `.venv-gpu/`, `DyVo/dyvo_data/*` (the untracked `codec/*.json*`, `*/docs_entities.jsonl`, qrels/topics/score files are re-downloadable from HF), `DyVo/outputs/`, `scratch/`, `results/`, `logs/*.log`.
> - Suggested: `git add DyVo/lsr scripts repro_codec *.md && git commit -m "CODEC repro: configs, run scripts, fixes, handoff" && git push` (then check `git status` shows only ignored/data files left). `results/` is git-ignored; copy `results/results.csv` by hand if wanted (it is regenerated by `04_collect_results.py`).

If something is missing after cloning, copy it directly from the old server:
```bash
rsync -av 24b4515@<old-host>:~/Dyvo_cs6101/{scripts,repro_codec,handoff.md,NOTES.md,ERRORS.md} ./
rsync -av 24b4515@<old-host>:~/Dyvo_cs6101/DyVo/lsr/ ./DyVo/lsr/
```

### 4.2 Python environment
```bash
uv python install 3.10                      # torch 2.2.2 pins need 3.10
bash scripts/big_pc/00_setup_env.sh --dry-run
bash scripts/big_pc/00_setup_env.sh --python $(uv python find 3.10)     # add --cuda cu118 for old drivers
source .venv-gpu/bin/activate
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # expect 2.2.2+cu121 True
bash scripts/verify.sh                       # offline gate, must print VERIFY PASSED (~1-2 min unloaded)
```
Key versions (exact list in `repro_codec/env/`, `requirements-lock.txt`): Python 3.10.21, torch 2.2.2+cu121, transformers 4.40.2, tokenizers 0.19.1, wandb 0.16.6. The lock is installed with `--no-deps` on purpose (ERRORS E5: a protobuf conflict pip rejects otherwise). `pip check` will list two known protobuf mismatches; ignore them.

### 4.3 Data: where everything comes from
| Data | Source | Command / destination | Size |
|---|---|---|---|
| Authors' DyVo data (CODEC training queries + Mixtral queries, MonoT5-3B scores, REL/GPT-4/Mixtral/BM25/LaQue/human entity files, `entity_list.json`, `dyvo_init/` checkpoint, LaQue embeddings) | HF **dataset** repo `lsr42/dyvo_data` (public) | `python scripts/big_pc/01_list_and_download_data.py --dry-run`, then `python scripts/big_pc/01_list_and_download_data.py --datasets codec --embeddings laque` → `DyVo/dyvo_data/` | ~17 GB for CODEC+LaQue (`--datasets codec,robust04,wapo` adds ~1 GB of Robust04/WaPo files that we do not use; skip them) |
| **CODEC document corpus** (729,824 docs) | HF dataset `macavaney/codec` (public, despite ir_datasets saying "on request") | see below → `~/.ir_datasets/codec/v1/comets_documents.jsonl` | 5.2 GB |

Note the raw command `huggingface-cli download lsr42/dyvo_data` from the upstream README is wrong (missing `--repo-type dataset`); use the script, which calls `hf download lsr42/dyvo_data --repo-type dataset --local-dir DyVo/dyvo_data --include <pattern>`.

CODEC corpus:
```bash
mkdir -p ~/.ir_datasets/codec/v1
curl -L https://huggingface.co/datasets/macavaney/codec/resolve/main/documents.jsonl.gz -o /tmp/documents.jsonl.gz
sha256sum /tmp/documents.jsonl.gz     # must be c567900f432622637677060b5d24808bc0e4b6b38806b045d8815a0bedadab90
gunzip -c /tmp/documents.jsonl.gz > ~/.ir_datasets/codec/v1/comets_documents.jsonl
wc -l ~/.ir_datasets/codec/v1/comets_documents.jsonl      # expect 729824
```
ir_datasets also needs the topics/qrels for `codec`; it fetches them itself on first use (small). The old server has an offline guard (`scripts/offline_guard/sitecustomize.py`, used by `verify.sh` only) that is not needed for real runs.

Alternative to re-downloading (only if the old server is reachable): `rsync -avP` the same folders (`DyVo/dyvo_data/`, `~/.ir_datasets/codec/`) and skip the internet. Do that only if the HF download is slow.

### 4.4 Check before spending GPU hours
```bash
python scripts/big_pc/02_check_data.py --experiments priority1
```
Must report **0 errors** (paths, RAM, corpus loads, qrels ids present in corpus (5917/5917), teacher scores non-negative, `entity_dim` matches). The "priority1" pair is the CODEC LSR-w and DyVo(REL) at L1 = 1e-5, which is exactly our target.

---

## 5. What to run

Everything is launched through `scripts/big_pc/03_train.sh <experiment> [--wait] [--dry-run] [-- hydra overrides]`. It skips finished runs (detects `DyVo/outputs/<exp>/*/test_result.json`), refuses double-starts, writes `logs/<exp>.log`, and sets `LSR_SCRATCH_DIR=./scratch`, `WANDB_MODE=offline`, `WANDB__SERVICE_WAIT=300`. It defaults to `BATCH_SIZE=16` (per-device); the old server used `BATCH_SIZE=4`.

Always run inside tmux so it survives disconnects: `tmux new -s dyvo`, run, detach with `Ctrl-b d`, reattach with `tmux attach -t dyvo`. (Runs are started with `nohup setsid`-style wrapping, but tmux is still the safest.)

Set once per shell:
```bash
export E=qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05
export W=qmlp_dmlm_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001
```

### 5.1 Recommended order (fastest path to the headline number)

**Step A: smoke check on the new machine (5 min).** `bash scripts/verify.sh` (already in §4.2).

**Step B: DyVo 20k re-test, no training needed (gets a first DyVo number in ~1 h).** Requires the 20k DyVo weights, copy them from the old server:
```bash
mkdir -p DyVo/outputs/${E}_course20k/09-29-2026
rsync -avP 24b4515@<old-host>:~/Dyvo_cs6101/DyVo/outputs/${E}_course20k/09-29-2026/model DyVo/outputs/${E}_course20k/09-29-2026/     # 510 MB
ln -sfn 09-29-2026 DyVo/outputs/${E}_course20k/final
bash scripts/big_pc/03_train.sh ${E}_course20k_eval_rel      # REL candidates test (1 step at lr 0, then official predict)
bash scripts/big_pc/03_train.sh ${E}_course20k_eval_gpt4
bash scripts/big_pc/03_train.sh ${E}_course20k_eval_mistral
```
These `*_eval_*` configs load `outputs/<train run>/final/model/{query,doc}_encoder/model.safetensors`, take one optimizer step at learning-rate 0 (weights unchanged, λ_ent unchanged), and then run the official `predict()`. **The 20k model was trained on the old server with batch 4×4; if you cannot reach the old server, retrain it (Step C, ~1–3 h on a dedicated A100-class GPU).**

**Step C: paper-length training, in this order**
```bash
# (1) LSR-w 100k
bash scripts/big_pc/03_train.sh ${W}_full100k --wait
# (2) DyVo REL 100k (needs >= 48 GB RAM free)
bash scripts/big_pc/03_train.sh ${E}_full100k --wait
# (3) inference-only rows on the trained DyVo model
ln -sfn "$(ls DyVo/outputs/${E}_full100k | head -1)" DyVo/outputs/${E}_full100k/final   # date dir that holds model/ + test_result.json
bash scripts/big_pc/03_train.sh ${E}_full100k_eval_gpt4        # HEADLINE ROW
bash scripts/big_pc/03_train.sh ${E}_full100k_eval_mistral
```
Both `full100k` YAMLs default to batch 4 × accum 4. On a big GPU: append `-- training_arguments.per_device_train_batch_size=16 training_arguments.gradient_accumulation_steps=1 loss.d_regularizer.T=50000` (and set `BATCH_SIZE=16`); **verify the resolved config keeps effective batch 16** (`repro_codec/resolved_configs/*.yaml` shows what was used; new resolved configs must be regenerated and saved, see §8).

Extra rows if time remains (cheap, inference-only): `${E}_course20k_eval_gpt4`, `_eval_mistral`, `${E}_init`, and the BM25 / LaQue / human candidate configs (`${E}_bm25`, `${E}_laquecand`, `${E}_human`; these train-config names are the *training* variants and would retrain, so to make them inference-only, write `*_eval_*` variants copying `..._full100k_eval_gpt4.yaml` with the base `defaults:` entry swapped, **not implemented yet**).

### 5.2 Resuming a half-trained run instead of restarting (optional, saves ~50 h of shared-GPU time)
If you can copy the newest checkpoint from the old server before it is shut off:
```bash
# LSR-w, newest checkpoint dir (>= checkpoint-45000; pick the highest number that exists)
rsync -avP 24b4515@<old-host>:~/Dyvo_cs6101/DyVo/outputs/${W}_full100k/09-29-2026/checkpoint-<N> \
      DyVo/outputs/${W}_full100k/09-29-2026/
# DyVo REL, newest checkpoint dir (>= checkpoint-20000)
rsync -avP 24b4515@<old-host>:~/Dyvo_cs6101/DyVo/outputs/${E}_full100k/09-29-2026/checkpoint-<N> \
      DyVo/outputs/${E}_full100k/09-29-2026/
# resume
bash scripts/big_pc/03_train.sh ${W}_full100k --wait -- resume_from_checkpoint=$PWD/DyVo/outputs/${W}_full100k/09-29-2026/checkpoint-<N>
```
⚠️ **Untested path.** `resume_from_checkpoint` is wired in `lsr/train.py` (`trainer.train(conf.resume_from_checkpoint)`), and `HFTrainer._load_from_checkpoint` is overridden to load only model weights via `from_pretrained`. Each checkpoint has `optimizer.pt`, `scheduler.pt`, `trainer_state.json` and `rng_state.pth`, so the HF Trainer restores the step counter/optimizer. **Before trusting it:** start it, confirm the log shows `Continuing training from checkpoint … global step N` and the first `loss_no_reg` is near the value before the checkpoint (≈ 0.22–0.23 for both models), and that `query #entities` / `doc #entities` are > 0 for DyVo. If it looks wrong, restart from scratch; do not hand-edit.

Also, if `resume_from_checkpoint` is ever given a path relative to `DyVo/`, remember `03_train.sh` `cd`s into `DyVo/`; use an absolute path.

### 5.3 Turn everything into the deliverables
```bash
python scripts/big_pc/04_collect_results.py      # -> results/results.csv (ours vs paper, diff; 46-row table, only finished ones filled)
python scripts/big_pc/codec_report.py            # -> repro_codec/REPORT.md
bash scripts/big_pc/codec_snapshot.sh            # refreshes repro_codec/{env,resolved_configs}: versions, pip freeze, hardware, local patch diff, data provenance
```
`repro_codec/` is small and **should be tracked in git** (it is currently untracked, see §4.1); commit it after each finished run so results are never only on the compute machine. Also copy each `test_result.json` (and the model dir for rows 1–2, ~0.5 GB) somewhere off the machine. `DyVo/outputs/` is git-ignored.

---

## 6. Time budget and fallback plan (deadline 3 Oct)

Measured on the **old shared** server: 3.7–4.0 s/optimizer-step for the 100k runs (GPU shared with other users and CPU-bound data loading; the entity model is memory-heavy). So 100k steps ≈ 100+ h there, which does **not** fit before 3 Oct. Measured for a 20k run there: LSR-w 5.2 h, DyVo 7.5 h. Test inference: 43–48 min per LSR-w model at eval batch 64 (DyVo tests are slower, as each doc also scores entities).

On a dedicated 24 GB+ GPU expect roughly 0.3–1 s/step, i.e. ~8–28 h per 100k-step run, plus 1–2 h per test. **Measure the first 500 steps and extrapolate before committing to 100k.** Decision rule:

| If the projected finish of `E_full100k` + its tests is… | Do |
|---|---|
| before ~2 Oct | Run rows 1–4 as planned (§5.1 step C). |
| after that | Present the **20k-step rows (5–7)** as the main reproduction (labelled "course budget", 20 % of paper steps) and report the 100k rows as "in progress / partial checkpoint". Do **not** claim paper-length results for runs that did not finish. |

Order matters: get **one complete DyVo(REL) → GPT-4 → Mixtral chain** before starting anything optional, because the GPT-4 row is the paper's headline.

Two RAM/GPU rules: never launch two DyVo processes together unless ≥ 90 GB RAM is free; and prefer the GPU with most free memory (`nvidia-smi`). `run_codec_phase2.sh` / `run_codec_fix.sh` automate the waiting on the shared server; on a private machine they are unnecessary, run `03_train.sh` directly.

---

## 7. Known bugs, pitfalls, and gotchas

1. **Stale-scratch crash (fixed in working tree, verify it is in git):** `evaluation_loop_iterative_vectorized` in `DyVo/lsr/trainer/__init__.py` used one `try` to delete both `queries.jsonl` and `docs/`; a missing queries file left a stale `docs/` dir and `docs_dir.mkdir()` raised `FileExistsError` at test time. This is exactly how the DyVo 20k test crashed after 7.5 h of training. The fix (independent cleanup + `mkdir(exist_ok=True)`) must be present *before* starting any DyVo run. The two 100k processes on the old server started before the fix, so they would have hit it; you will not, provided you use the fixed file. If a test crashes anyway, the trained weights are in `DyVo/outputs/<exp>/<date>/model/`; re-test via an `*_eval_*` config rather than retraining.
2. **`FileExistsError`/stale dirs in `scratch/outputs/<exp>/…/tmp_eval`**: safe to delete the folder `scratch/outputs/<exp>` if a run was aborted.
3. **CUDA OOM** only happened because of other users on the shared GPU. Use a free GPU; use eval batch 64 on a big card, lower it (`training_arguments.per_device_eval_batch_size=8`) if it OOMs.
4. **wandb:** pinned `0.16.6` (newer removes `start_method`, ERRORS E2); runs are offline; service timeout raised to 300 s.
5. **`loss.normalize: "std"` is actually L2 normalisation and requires non-negative teacher scores** (NaN otherwise, ERRORS E1). `02_check_data.py` checks this.
6. **`robust04/queries_entities_gpt4.jsonl` is misaligned** upstream; irrelevant for us (Robust04 skipped). CODEC GPT-4/Mixtral files are fine (test queries 42/42 present).
7. **Harmless warning:** `faiss.swigfaiss_avx2` import; falls back to the generic build.
8. **The old docs' timing (0.2–0.4 s/step, 6–12 h/run) was an estimate and did not hold on the shared server**; use §6.
9. **Old logs inflate the entity-collapse check:** for LSR-w, `query/doc #entities` is 0 by design (no entities). The check applies to DyVo only.

## 8. Health checks for every DyVo run (from paper App. A.2)
1. `query #entities` and `doc #entities` in the log must stay > 0 (0 = entity collapse). Healthy so far: ≈ 0.9 and ≈ 20.
2. `loss_no_reg` is finite and falling (≈ 0.40 → 0.22–0.23 on CODEC).
3. Result within ~1–2 nDCG@10 points of the paper; a gap of 10+ means something is mis-wired. Expected: DyVo(REL) ≈ 53.4; GPT-4 candidates ≈ 56.5 (large gain); Mixtral ≈ 54.9. **The ordering LSR-w < DyVo(REL) < Mixtral < GPT-4 is the scientific claim; report it even if absolute numbers are 1–2 points off.**
4. Record for the report, per run: exact command, resolved config (`repro_codec/resolved_configs/`), seed 42, dependency versions, wall-clock, GPU, whether it is 20k or 100k steps.

## 9. Checklist for the next person (copy and tick)

- [ ] Committed + pushed all code from the old server (§4.1 warning) and confirmed `git log` on the new machine shows it
- [ ] New machine: Python 3.10 env, torch CUDA visible, `verify.sh` → VERIFY PASSED
- [ ] `lsr42/dyvo_data` (CODEC + LaQue, ~17 GB) downloaded; CODEC corpus at `~/.ir_datasets/codec/v1/comets_documents.jsonl` with the right sha256 and 729,824 lines
- [ ] `02_check_data.py --experiments priority1` → 0 errors
- [ ] Copied off the old server (if reachable): 20k DyVo `model/` (510 MB); newest `checkpoint-*` of both 100k runs (1.5 GB each); `results/results.csv`; `logs/` (15 MB)
- [ ] Decide budget from the first 500 steps (§6)
- [ ] DyVo 20k tested with REL → GPT-4 → Mixtral candidates
- [ ] LSR-w 100k done + tested
- [ ] DyVo REL 100k done + tested; GPT-4 and Mixtral inference rows done
- [ ] `04_collect_results.py`, `codec_report.py`, `codec_snapshot.sh` run; `repro_codec/` committed and pushed
- [ ] Midterm write-up filled from `repro_codec/MIDTERM.md` with: results table (ours vs paper), scope limits (§2), deviations (§2), the bug list (§7)

## 10. Open questions / not verified
- The **resume-from-checkpoint** path (§5.2) has not been exercised.
- **Effective-batch/regulariser-T equivalence** when switching to per-device batch 16 (§2 deviation 2) is reasoned from the config comments, not tested. When in doubt keep the 4 × 4 setup, which is what the smoke/20k runs used.
- DyVo test time at full corpus with a dedicated GPU is unmeasured (only LSR-w's 48 min).
- The numbers in §3 come from the old server's logs as of ~15:55 IST on 2026-09-29 and will have moved on if that server is still running.
