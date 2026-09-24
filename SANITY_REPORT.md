# DyVo build-phase sanity report

> **No real model and no real data were used anywhere in this report.** Every model is a tiny, randomly
> initialized DistilBERT (dim 32, 1 layer), every entity table is random, and the corpora are 20 synthetic
> documents. **None of the numbers below say anything about retrieval quality.** They only show that the code
> runs end to end and behaves as the paper describes. Real results come from the big-PC runs (`BIG_PC_HANDOFF.md`).

Machine: Apple M4, 16 GB RAM, CPU only (no CUDA), Python 3.10.20, torch 2.2.2 (CPU), transformers 4.40.2.
Gate: `bash scripts/verify.sh` → **exit 0, `VERIFY PASSED` (47–124 s depending on machine load)**, fully offline (see "Enforcement").
Upstream code: `thongnt99/DyVo` @ `1b30d6c`, extended in place (every edit is marked `# DYVO-REPRO`).

## Enforcement of the resource rules
- `verify.sh` exports `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=""`.
- It also puts `scripts/offline_guard/sitecustomize.py` on `PYTHONPATH`. That makes every outbound socket or DNS lookup in every Python process raise. The gate's first step proves it: `ok: [offline guard] DNS lookup blocked during verification: 'huggingface.co'`.
- Smoke configs set `use_cpu: True`, so HF Trainer does not pick up Apple MPS.
- Downloads made during the build: only the 16 allowed query-entity files (2.9 MB total), plus a file *listing* of `lsr42/dyvo_data`.

## Level 0: static checks
| Check | Command | Result | Evidence |
|---|---|---|---|
| Compile every `.py` | `find DyVo scripts -name '*.py' \| xargs python -m py_compile` | PASS | `py_compile ok: 130 files` |
| Shell syntax | `bash -n scripts/**/*.sh` | PASS | `bash -n ok` |
| Imports | every submodule of `lsr.models/losses/datasets/trainer` + entity modules | PASS | needed fix: `lsr/losses/__init__.py` imported 3 deleted modules |
| Config composition | `bash scripts/check_configs.sh` (`python -m lsr.train +experiment=<name> --cfg job`) | PASS | `configs composed: 73 passed, 0 failed` (43 official + 28 new Table 2/3 + 2 smoke). Initially 32/43: 11 official configs referenced a missing `dataset/trec_robust04_fold1` (added) |
| Entity width vs config | `python scripts/check_entity_dims.py` | PASS | `53 DyVo configs checked, 0 mismatches`. Caught the upstream Wiki2Vec bug (`entity_dim: 768` for a 300-d table) |

## Level 1: unit tests (tiny assets)
`python -m pytest DyVo/tests -q` → **61 passed** (12 files).

| File | What it proves |
|---|---|
| `test_entity_emb.py` | IDs offset by 30522 give the right rows; shape `(B, n_ent, dim)`; the table is not a parameter, is not saved, and gets no gradient |
| `test_dyvo_head.py` | Entity weights of **both** encoders equal a hand-computed `max_j log1p(relu(λ·e_i·h_j))·mask` within 1e-5. Masked slots are 0, all weights ≥ 0, output IDs = word IDs ++ entity IDs. The MLP query encoder does not expand; the doc word part equals the SPLADE max-pool reference |
| `test_projection.py` | Entity dim 16 creates a trainable 16→32 projection that receives gradients; dim 32 creates none |
| `test_sparse_dot.py` | `sparse_dot_product` equals a scatter-to-dense reference (including duplicate IDs and real encoder outputs); word 5 never matches entity 5 (30527) |
| `test_lambda_ent.py` | `ent_weight` is an `nn.Parameter` initialized from config (0.05) and gets a gradient; entity weights are non-zero at init |
| `test_collator.py` | `DynamicDataCollator`: per-group padding, masks, 30522 offset, zero-entity texts get one masked slot, prediction batches |
| `test_candidates.py` | `BERTCandidate`: dedup, int IDs, masks, unknown/empty → `[]`, reads the authors' LLM-file format |
| `test_entity_zero.py` | Real entity 0 vs a padded ID-0 slot: padded weight is exactly 0 and contributes 0 to the score |
| `test_loss.py` | `EntityDistilKLLoss` is finite; gradients reach both backbones, both λ's and the projection; L1 skips entities. **Documents a pitfall:** `normalize: "std"` gives NaN when teacher scores are negative (E1) |
| `test_checkpoint_load.py` | `DualSparseEncoder` loads the safetensors init checkpoints. Only the tied `vocab_projector.weight` is "missing" (it stays tied); a marker value proves the load happened. LSR-w encoders load the same files, ignoring `ent_weight`/`tok_weight` |
| `test_merge_candidates.py` | Train IDs keep REL, test IDs take the source, missing IDs fall back to REL, output reads through `BERTCandidate`. On the real files, `--source human` reproduces the authors' merged human file **exactly** |
| `test_llm_entities.py` | Prompt A.2 is verbatim. The parser handles JSON, prose, code fences, Python lists, truncated output, bullets, double-encoding and duplicates, and recovers the exact `entity_names` of all 6 authors' GPT-4/Mixtral files. Case-insensitive name→ID mapping with OOV drop; mock end-to-end; remap mode; the OpenAI backend refuses to run without a key |
| `test_big_pc_helpers.py` | All 46 paper experiments have configs and map onto all 16 paper table rows |

## Level 2: behavioral invariants
- **Overfit one batch** (`python DyVo/sanity/overfit_one_batch.py`): 2 queries × (pos, neg) through the real collator and candidate retriever, Adam lr 1e-3, 100 steps, softmax teacher. Pass threshold: final ≤ 0.5 × initial and some entity weights still non-zero.
  **Result: KL 0.40172 → 0.00028 (ratio 0.001); non-zero entity weights 15/15 → PASS.** With `--normalize std` (the configs' setting): 0.58127 → 0.00017, PASS.
- **Collapse probe** (paper App. A.2), 300 steps each:

  | Run | Loss start → end | Live entity weights | Full collapse? |
  |---|---|---|---|
  | λ=1.0 frozen | 0.256 → 0.006 | 15 → 12 | no |
  | λ=0.05 trainable (paper) | 0.402 → 0.00007 | 15 → 15 (λ drifts to ≈0.066) | no |
  | λ=1.0 frozen, std | 0.0001 → 0.0000 | 15 → 15 | no |
  | λ=0.05 trainable, std | 0.581 → 0.00004 | 15 → 15 | no |

  Honest reading: the paper's collapse does **not** reproduce at toy scale. λ=1.0 frozen is the only setting where entity weights start to die, which points in the paper's direction but is weak evidence. Watch `query #entities` / `doc #entities` in the first big-PC run.
- Other invariants are covered by the Level 1 tests above: weights ≥ 0, masked entities exactly 0, word IDs < 30522 ≤ entity IDs, sparse dot equals the dense reference, and gradients reach both encoders, the projection and `ent_weight`.

## Level 3: end-to-end smoke (official entry point)
`python -m lsr.train +experiment=smoke_dyvo` and `+experiment=smoke_lsrw`. Both mirror the real CODEC configs (same classes, loss, collator, checkpoint loading) with tiny paths, 20 steps, batch 2, CPU.

| Run | Exit | Artifacts | Metrics (**SANITY ONLY, meaningless**) |
|---|---|---|---|
| smoke_dyvo | 0 (~6 s) | `checkpoint-10/20`, `model/{query,doc}_encoder`, `loss.pt`, `test_run.trec` (100 lines), `test_result.json`; mid-training eval ran at step 10 and 20 | nDCG@10 0.6719, nDCG@20 0.7569, R@1000 1.0; train log `query #entities 1.8`, `doc #entities 3.0` |
| smoke_lsrw | 0 (~5 s) | same | nDCG@10 0.3750, nDCG@20 0.5147, R@1000 1.0; `query #entities 0.0` |

Seeds are fixed (HF seed 42, scripts seed 0), and numbers are identical across verify runs.

## Level 4: big-PC scripts
All 8 files in `scripts/big_pc/` pass `bash -n` / `py_compile` and `--dry-run` inside `verify.sh`, with the network guard on:
`00_setup_env.sh`, `01_list_and_download_data.py`, `02_check_data.py`, `03_train.sh`, `04_collect_results.py`, `05_generate_entities.sh`, `experiments.py`, `run_priority.sh`.
`run_priority.sh --dry-run` plans 46 training runs in 5 priority groups. `02_check_data.py` and `04_collect_results.py` were also run for real here; they only read local files and reported what is missing, as expected.

## Upstream issues found and fixed (details in NOTES.md)
1. `lsr/losses/__init__.py` imported 3 deleted modules, so no config could train from a fresh clone.
2. `TransformerMLPeMLMConfig` never stored `entity_cls`, so the DyVo query encoder's forward raised `AttributeError`.
3. 11 configs referenced a missing `dataset/trec_robust04_fold1` → added (ir_datasets fold 1).
4. The Wiki2Vec configs' `entity_dim: 768` doesn't match the 300-d table, so the first step would fail with a shape error → fixed.
5. Eval wrote to the authors' cluster path `/scratch-shared/$USER` → now honors `LSR_SCRATCH_DIR`.
6. `wandb.Settings(start_method=...)` fails on current wandb → pinned wandb 0.16.6.

Findings that were not fixed but are flagged for the big PC:
- `robust04/queries_entities_gpt4.jsonl` looks like it uses a different entity ID space (re-mapped by `run_priority.sh`).
- `normalize: "std"` needs non-negative teacher scores (checked by `02_check_data.py`).
- Configs inherit batch 128 while the paper says 16 (`03_train.sh` passes 16).
- The DyVo collator truncates queries at `d_max_length`.
