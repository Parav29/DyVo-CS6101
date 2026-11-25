# Follow-up experiments F1–F3: design declared before running

Declared 2026-11-25 11:37 IST, before any F1–F3 number was computed. This file is not edited after the runs; results go to `result.md` (repo root).

Frozen model: DyVo full100k (as in E1/E2). Test set: 42 CODEC queries, full corpus. Test judgments are used only for scoring and for topic-disjoint selection (below), never to pick a setting that is then scored on the same topics.

## Configurations

Each run is (query base list, E1 additions, document index):

- **Base list** b ∈ {REL, GPT-4, Mixtral}: the released query entities.
- **E1 additions**: add the top-K entities chosen by the E1-occ v2 selector (`scratch/ext/e1/results_prf/queries_E1-occ_k{K}.jsonl`, trained on the linked index) to the base list, K ∈ {0, 1, 3, 5, 10, 20}. K = 0 means the base list alone. Disclosure: we use the linked-index selectors because they are E1's main configuration (§7.3). We already know from §7.5 that the selectors recomputed on the repaired index scored lower.
- **Document score** = S_linked + α·(S_repaired − S_linked), α ∈ {0, 0.25, 0.5, 0.75, 1}. S_repaired comes from the E2 v2 grounded c=5 index (`scratch/ext/index_grounded_v2`). α = 0 is the original DyVo index; α = 1 is the E2 index.

Query vectors: each test query is encoded once (batch 1) with the union of every candidate it could receive, as in `e1_run.py` (exact, §3.4 of extension1.md).

## Systems

| System | Base | K | α | Compared with |
|---|---|---|---|---|
| **DyVo baseline** | REL | 0 | 0 | – |
| E1 alone (E1-occ v2) | none | CV over {1,3,5,10,20} | 0 | DyVo baseline |
| E2 alone (grounded v2 c=5) | REL | 0 | 1 | DyVo baseline |
| **F1** REL + E1 | REL | CV over K | 0 | DyVo baseline |
| **F2** REL + interpolated E2 | REL | 0 | CV over α | DyVo baseline |
| **F1+F2** | REL | CV over K × α | | DyVo baseline |
| F1 / F2 / F1+F2 on GPT-4 | GPT-4 | as above | | GPT-4 (K=0, α=0) and DyVo baseline |
| F1 / F2 / F1+F2 on Mixtral | Mixtral | as above | | Mixtral (K=0, α=0) and DyVo baseline |

CV = the nested topic-disjoint selection of `e1_cvk.py`. The setting is chosen on a random half of the topics by mean nDCG@10 (ties go to the smaller K, then smaller α) and applied to the other half, then the halves swap. This is repeated over 50 random splits. We report the mean ± sd over splits and the fraction of splits that beat the comparison. Paired tests use split 0.

## Statistics (F3)

- Paired randomization test (10,000 permutations) and 95% bootstrap CI on per-query nDCG@10 **and** R@1000.
- **Primary family** (Holm, separately per metric): the five REL-side systems (E1 alone, E2 alone, F1, F2, F1+F2) vs the DyVo baseline.
- **Secondary family** (Holm, per metric): the six GPT-4 / Mixtral F-systems vs their own base list.
- A gain is called an improvement only if p_Holm < 0.05. Otherwise it is reported as a non-significant difference, with its sign stability across splits.

Everything above is reported, whatever the result.
