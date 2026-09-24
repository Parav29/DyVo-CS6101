# Traceability: prompt file + paper → implementation

Sources:
- `DYVO_IMPLEMENTATION_PROMPT.md`: the task spec (Sections 0–8 plus Appendices A–B).
- `cs6101_DyVo.pdf`: *DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities* (Nguyen et al., EMNLP 2024), 17 pages.

Both were read in full at the start of the build. The paper text was extracted with `pypdf` to quote equations, hyperparameters and Prompt A.2 verbatim.

## A. Paper → code
| Paper element | Where it is implemented / verified |
|---|---|
| Eq. 1: sparse dot-product score | `DyVo/lsr/losses/entity_distil_kl_loss.py::sparse_dot_product`; tested in `DyVo/tests/test_sparse_dot.py` against a dense reference |
| Eq. 2: MLM doc encoder `max_j log1p(ReLU(e_i·h_j))` | `DyVo/lsr/models/mlm_emlm.py`; `tests/test_dyvo_head.py::test_doc_word_weights_match_splade` |
| Eq. 3: MLP query encoder (no expansion) | `DyVo/lsr/models/mlp_emlm.py`; `tests/test_dyvo_head.py` (word IDs = input IDs, masking) |
| Eq. 4: DyVo head `λ_ent · max_j log1p(ReLU(e_ent·h_j))` | both `*_emlm.py` files. Hand-computed check in `tests/test_dyvo_head.py`. The λ-inside-ReLU difference is logged in `NOTES.md` (Deviations) |
| λ_ent trainable, init 0.05 (§3.2) | `ent_weight` `nn.Parameter`; `entity_weight: 0.05` in configs; `tests/test_lambda_ent.py` |
| Eq. 5: joint word + entity score | ID concatenation (entity k → 30522 + k); `tests/test_entity_zero.py`, `test_sparse_dot.py` |
| §3.3: frozen entity embeddings + projection when sizes differ | `lsr/models/entity_emb.py`, `entity_proj`; `tests/test_entity_emb.py`, `test_projection.py` |
| §3.3: entity candidate retrieval (REL, BM25, LaQue, LLMs) | `lsr/entity/bert_candidates.py`; `lsr/entity/merge_candidates.py` (test-query candidates from the source, REL for training); 28 Table 2/3 configs |
| Prompt A.2 (few-shot generative entity retrieval) | `lsr/entity/generate_entities_llm.py::PROMPT_A2` (verbatim); `tests/test_llm_entities.py::test_prompt_is_paper_prompt_a2` |
| §4: training (100k steps, batch 16, lr 5e-7, fp16, KL from MonoT5-3B, L1 on doc words ∈ {1e-3, 1e-4, 1e-5}, no L1 on entities/queries) | official experiment YAMLs; `scripts/big_pc/03_train.sh` (adds batch 16 + fp16); `tests/test_loss.py::test_l1_skips_entities_when_reg_ent_false` |
| §4: evaluation nDCG@10, nDCG@20, R@1000 (ir_measures) | `lsr/trainer` predict loop; `scripts/big_pc/04_collect_results.py` |
| Tables 1–3 (reference numbers) | `scripts/big_pc/experiments.py::PAPER`; all 16 rows mapped to 46 runs (`tests/test_big_pc_helpers.py`) |
| App. A.2: entity collapse | `DyVo/sanity/overfit_one_batch.py --lambda-init 1.0 --no-train-lambda --probe`; results in `SANITY_REPORT.md` (not reproduced at toy scale; reported honestly) |
| Table 4 examples ("NFTs" → Non-fungible token, etc.) | used as realistic entity names in `smoke_assets/build_assets.py` and in the LLM-parser tests |

## B. Prompt file → deliverables
| Prompt section | Deliverable |
|---|---|
| §0 Resource rules | No weights, datasets or LLM calls; only the 16 allowed files (<1 MB each). Offline enforced by `scripts/verify.sh` + `scripts/offline_guard/sitecustomize.py` |
| §1 Definition of done | All 30 tasks `[x]` in `PROGRESS.md`; `verify.sh` exit 0; dry-runs pass; both reports written |
| §2 Loop protocol | `PROGRESS.md`, `ERRORS.md` (E1–E4, all fixed), `NOTES.md`; one git commit per iteration (iter 0–10) |
| §3 / §3b What DyVo is; LLM facts | confirmed and extended in `NOTES.md` ("LLMs and API keys", with the S5.0 stats table and the Robust04 GPT-4 ID-space finding) |
| §4 Official-repo map | cloned to `DyVo/`; code ↔ paper map in `NOTES.md` |
| §5 Sanity-check design (tiny assets, synthetic data, 5 levels) | `smoke_assets/build_assets.py`, `smoke_data/build_data.py`, Levels 0–4 in `verify.sh` |
| §6 S0: scaffolding | `.claude/commands/dyvo-{iter,fix,verify,status}.md`, state files, environment recorded |
| §6 S1: code + environment | `.venv` (Python 3.10, CPU torch 2.2.2, transformers 4.40.2), `requirements-lock.txt` |
| §6 S2: tiny assets + unit tests | `scripts/check_configs.sh` (73/73 compose), `DyVo/tests/` (61 tests) |
| §6 S3: behavioral checks | `DyVo/sanity/overfit_one_batch.py` (overfit PASS, collapse probe) |
| §6 S4: end-to-end smoke | `lsr/configs/experiment/smoke_{dyvo,lsrw}.yaml`, `scripts/verify.sh` |
| §6 S5: LLM entity candidates | HF listing + stats, `merge_candidates.py`, 28 configs, `generate_entities_llm.py`, tests |
| §6 S6: big-PC scripts | `scripts/big_pc/00…05, run_priority.sh, experiments.py` (all `--dry-run`) |
| §6 S7: reports | `SANITY_REPORT.md`, `BIG_PC_HANDOFF.md` (includes "Do I need LLMs or API keys?") |
| §7 Reference numbers | `experiments.py::PAPER` (used only for comparison in `04_collect_results.py`) |
| §8 Engineering standards | every upstream edit marked `# DYVO-REPRO` and logged in `NOTES.md`; seeds fixed |
| Appendix A: slash commands | `.claude/commands/` (4 files, verbatim) |
