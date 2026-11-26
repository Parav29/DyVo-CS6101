# DyVo (CS6101): reproduction + two extensions

Course project. We reproduced **DyVo** ("Dynamic Vocabularies for Learned Sparse Retrieval with Entities", EMNLP 2024; [paper](https://aclanthology.org/2024.emnlp-main.45/)) on the CODEC collection, then built and tested two extensions on top of the frozen reproduced model.

**This repository holds code, write-ups and small result summaries only.** Data, model checkpoints, the Python environment and large indexes (tens of GB) are deliberately not included; they are listed in `.gitignore`.

## Read in this order
| # | File | What you learn |
|---|---|---|
| 1 | `REPORT.md` | Reproduction: method, setup, results vs the paper, deviations |
| 2 | `EXTENSIONS.md` | Extension proposals: query entity selection (E1) and document entity repair (E2) |
| 3 | `extension1.md` | E1: method, results (§7), conclusions (§8) |
| 4 | `extension2.md` | E2: method, results (§7), blinded review, conclusions (§8) |
| 5 | **`result.md`** | **Extensions vs baseline DyVo, with significance tests: the main results table** |

Result files are in `scratch/ext/` (test metrics, cross-validation, review judgements; follow-up design in `followup_design.md`).

## Code map
- `DyVo/` – the authors' code with our fixes (marked `DYVO-REPRO`).
- `scripts/big_pc/` – reproduction: training supervisors, fast evaluator (`fast_eval.py`), data setup and checks.
- `scripts/ext/` – extensions: `common.py` (model, exact scoring, statistics), `dump_index.py`; E1: `e1_pool.py`, `e1_prf.py`, `e1_labels.py`, `e1_run.py`, `e1_cvk.py`, `e1_latency.py`; E2: `e2_alias.py`, `e2_mentions.py`, `e2_dense.py`, `e2_build.py`, `e2_make_index.py`, `e2_run.py`, `e2_diag.py`; follow-ups: `f_followups.py`; pipeline: `run_after_delta.sh`.

## Headline results (CODEC, 42 test queries; details and caveats in `result.md`)
| System | nDCG@10 | R@1000 |
|---|---:|---:|
| Baseline DyVo (REL entities) | 53.67 | 70.82 |
| Extension 1 alone | 54.75 | 73.21 (significant, p_Holm = 0.002) |
| Extension 2 alone | 54.14 | 70.99 |
| Both extensions | 55.03 | 73.32 (significant, p_Holm = 0.002) |

nDCG@10 gains are consistent across topic splits but not statistically significant with 42 topics; the recall gains are. The extensions' original hypotheses (utility-trained selection; grounded document repair vs controls) were not supported; see `extension1.md` §8 and `extension2.md` §8.

## Rerunning
Needs the authors' data, the reproduced checkpoint (`DyVo/outputs/...full100k`) and the document index built by `scripts/ext/dump_index.py`. Run order is in `scripts/ext/run_after_delta.sh` and the progress tables at the top of `extension1.md` / `extension2.md`.
