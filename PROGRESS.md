# PROGRESS

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!] BLOCKED-EXTERNAL` / `[!] BLOCKED-INTERNAL`

## Phase S0: Scaffolding
- [x] S0.1 PROGRESS.md / ERRORS.md / NOTES.md, git init
- [x] S0.2 Slash command files (.claude/commands/dyvo-*.md)
- [x] S0.3 Environment recorded in NOTES.md

## Phase S1: Code + environment
- [x] S1.1 Clone thongnt99/DyVo into ./DyVo
- [x] S1.2 Python venv, CPU torch, deps, requirements-lock.txt
- [x] S1.3 Code ↔ paper map + known issues in NOTES.md

## Phase S2: Tiny assets + unit tests (Levels 0–1)
- [x] S2.1 DyVo/smoke_assets/build_assets.py
- [x] S2.2 scripts/check_configs.sh (43 real configs compose)
- [x] S2.3 Unit tests in DyVo/tests/

## Phase S3: Behavioral checks (Level 2)
- [x] S3.1 DyVo/sanity/overfit_one_batch.py
- [x] S3.2 Collapse probe (--lambda-init 1.0 --no-train-lambda)

## Phase S4: End-to-end smoke (Level 3)
- [ ] S4.1 Synthetic data in DyVo/smoke_data/
- [ ] S4.2 smoke_dyvo.yaml / smoke_lsrw.yaml run through lsr.train
- [ ] S4.3 scripts/verify.sh

## Phase S5: LLM entity candidates
- [ ] S5.0 HF listing + small query-entity files + stats table
- [ ] S5.1 lsr/entity/merge_candidates.py
- [ ] S5.2 Table 2 / Table 3 experiment configs
- [ ] S5.3 tests/test_merge_candidates.py
- [ ] S5.4 (optional) lsr/entity/generate_entities_llm.py
- [ ] S5.5 (optional) tests/test_llm_entities.py

## Phase S6: Big-PC scripts (write + dry-run only)
- [ ] S6.1 00_setup_env.sh
- [ ] S6.2 01_list_and_download_data.py
- [ ] S6.3 02_check_data.py (+ S6.6b robust04 docs.tsv check)
- [ ] S6.4 03_train.sh
- [ ] S6.5 04_collect_results.py
- [ ] S6.6 05_generate_entities.sh (optional)
- [ ] S6.7 run_priority.sh

## Phase S7: Reports
- [ ] S7.1 SANITY_REPORT.md
- [ ] S7.2 BIG_PC_HANDOFF.md
- [ ] S7.3 Final verify.sh green
