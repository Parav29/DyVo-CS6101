# DyVo (CS6101): reproduction + two extensions

Course project. We reproduced **DyVo** ("Dynamic Vocabularies for Learned Sparse Retrieval with Entities", EMNLP 2024; paper: `cs6101_DyVo.pdf`) on the CODEC collection, then built and tested two extensions on top of the frozen reproduced model.

**This repository holds code, write-ups and small result summaries only.** Data, model checkpoints, the Python environment and large indexes (tens of GB) are deliberately not included; they are listed in `.gitignore`.

## Read in this order
| # | File | What you learn |
|---|---|---|
| 1 | `project_context.md` | What DyVo is, what we reproduced, hardware, data layout |
| 2 | `repro_codec/REPORT.md`, `repro_codec/MIDTERM.md`, `final_submission.md` | Reproduction results vs the paper (CODEC nDCG@10 etc.), deviations |
| 3 | `EXTENSIONS.md` (+ `DyVo-final-extension-plan.md`, `DyVo-research-council-report.md`) | The plan: Extension 1 (query entity selection) and Extension 2 (document entity repair) |
| 4 | `extension1.md` | E1: method, assumptions, challenges, all results (§7), conclusions (§8) |
| 5 | `extension2.md` | E2: method, results (§7.1 v1, §7.3 v2), blinded review, conclusions (§8) |
| 6 | **`result.md`** | **Extensions vs baseline DyVo, with significance tests: the main results table** |
| 7 | `scratch/ext/followup_design.md` | Follow-up experiment design, written before the runs were executed |

## Code map
- `DyVo/` – the authors' code with our fixes (marked `DYVO-REPRO`).
- `scripts/big_pc/` – reproduction: training supervisors, fast evaluator (`fast_eval.py`), report generation.
- `scripts/ext/` – extensions: `common.py` (model, exact scoring, statistics), `dump_index.py`; E1: `e1_pool.py`, `e1_prf.py`, `e1_labels.py`, `e1_run.py`, `e1_cvk.py`, `e1_latency.py`; E2: `e2_alias.py`, `e2_mentions.py`, `e2_dense.py`, `e2_build.py`, `e2_make_index.py`, `e2_run.py`, `e2_diag.py`; follow-ups: `f_followups.py`; pipeline: `run_after_delta.sh`.
- `scratch/ext/**.json` – small result summaries (per-method test metrics, CV results, review judgements).

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
