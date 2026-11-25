# Extension 1: utility-trained query entity selection

**Status:** complete (23–25 Nov 2026). Conclusions in §8.
Plan sources: `EXTENSIONS.md` §Extension 1, `DyVo-final-extension-plan.md` (E1), `DyVo-research-council-report.md` §9A, §11.

## 1. What we are testing

DyVo's query entity head only weights the candidates it is given; it never searches the catalog. Candidate generators (REL linking, BM25, LaQue, LLMs) are optimised for linking or similarity, not for ranking documents. We train a small selector that reads a cheap candidate pool and keeps the K entities most likely to **help rank relevant documents** under the frozen DyVo model and the fixed document index.

```text
Original:  query -> candidate generator -> entity IDs -> DyVo query entity head -> sparse retrieval
Ours:      query -> cheap pool (REL U BM25-title U LaQue) -> selector -> top-K IDs -> same DyVo head -> same index
```

Unchanged: DyVo Transformer, word branch, entity head, LaQue embeddings, document index (all from the reproduced DyVo full100k run).

**Research question (RQ1):** with the same pool, the same K and the same frozen model, does a selector trained on document-ranking utility beat generic candidate ranking and teacher imitation?

## 2. Progress log

| Date | Step | Status |
|---|---|---|
| 23 Nov | Read the plan, council report and code; froze baseline = DyVo full100k (`DyVo/outputs/..._entw_0.05_full100k/10-01-2026/model`) | done |
| 23 Nov | `scripts/ext/common.py`: model loading, exact sparse scoring, metrics, paired randomization test, bootstrap CI, Holm correction | done |
| 23 Nov | `scripts/ext/dump_index.py`: encode the corpus once and save all 729,737 sparse document vectors (same model, collator, fp16, doc batch 16 and order as the reproduction) | done; reproduces REL nDCG@10 53.69 vs 53.70 (§4.3) |
| 23 Nov | Diagnostic: how many candidates have any document posting | done (§4.1) |
| 23 Nov | `scripts/ext/e1_pool.py`: candidate pool for 9,000 training + 42 test queries | done (~40 candidates/query) |
| 23 Nov | `scripts/ext/e1_labels.py`: features + 4 label types from training data only | done (349,219 rows) |
| 23 Nov | `scripts/ext/e1_run.py`: train selectors, pick K on training-dev, test, statistics | **done: v1 results in §7.1** |
| 23 Nov | Pool-coverage diagnostic → corpus-feedback (PRF) source added (`e1_prf.py`, "E1 v2") | done (§7.2) |
| 23 Nov | Nested topic-disjoint K selection over 50 splits (`e1_cvk.py`) | done (§7.3) |
| 23 Nov | Latency measurement (`e1_latency.py`) | done (§7.4) |
| 23 Nov | E1 + E2 combination: PRF pool, labels and selectors recomputed on the repaired (E2 v2 grounded) index (plan step 8) | done (§7.5) |
| 25 Nov | Session cut off by a server outage after the 23 Nov runs finished; §7.5 and conclusions written up from the logs and `results_prf_g2/` | done |
| 25 Nov | Follow-ups F1 (base list + E1 entities), F2 (linked/repaired score interpolation), F3 (R@1000 tests); design declared first (`scratch/ext/followup_design.md`), `scripts/ext/f_followups.py` | done: `result.md` |

## 3. Method details

### 3.1 Candidate pool (`e1_pool.py`)
Pool(q) = REL(q) ∪ BM25-title top-20 ∪ LaQue top-20, about 40 candidates per query.
- **REL**: the authors' linked query entities (available for 5,681 of the 9,000 training queries and all 42 test queries).
- **BM25-title**: our own BM25 (k1 = 0.9, b = 0.4) over Wikipedia titles, stopwords removed.
- **LaQue**: the authors' LaQue query encoder `lsr42/distilbert-base-uncased-dense-laque` (mean pooling) dotted with the LaQue entity table, as in the authors' `lsr/entity/retrieve_entity_with_laque.py`. Check: on the full catalog our implementation returns exactly the authors' released LaQue top-20 for all 42 test queries (100% overlap).
- Both generators search only the **indexable catalog**: the 685,656 entities (of 5,277,580) that have at least one linked document in CODEC. An entity with no posting contributes exactly 0 to every document score under the fixed index, so this loses nothing.
- For every pool member we store s_e(q), the weight the frozen DyVo query head gives it.

### 3.2 Supervision (`e1_labels.py`, training queries only)
For each training query: the positive document (`codec_mistral_qrels.json`) and about 100 MonoT5-3B-scored documents (`codec_monot5_3b_scores.json`). With the frozen model and fixed index, entity e contributes c_e(d) = s_e(q)·s_e(d) to the score of d. Four labels, same features and model:

| Label | Definition | Role |
|---|---|---|
| `imit` | e is in the REL list | teacher-list imitation control |
| `occ` | e has a posting in the positive document | DREQ-style document-label control |
| `pair` | s_e(q)·(s_e(d+) − mean s_e(d−)), d− = teacher score < 0.5 | analytic pairwise-utility control |
| `loo` | Σ_d t(d)·[log softmax(S)(d) − log softmax(S − c_e)(d)], t = normalised teacher scores, S = word + all pool entities | **E1: listwise leave-one-out utility** |

### 3.3 Selector (`e1_run.py`)
- MLP 19 → 64 → 64 → 1 over 19 features. None of them identifies the entity, so the selector cannot memorise IDs. Features: DyVo weight, raw logit, LaQue and BM25 scores and ranks, source flags, number of sources, log document frequency, IDF, catalog popularity rank, title/query token overlap, title length, rank of DyVo weight in the pool, scores relative to the pool maximum, pool size.
- Loss: ListNet, where the target is each label's positive part normalised within the query. Same loss, 8 epochs, Adam 1e-3 and seed 0 for every label.
- Data split: 8,000 training queries to fit, 1,000 held-out training queries as dev (seed 0). K ∈ {1, 3, 5, 10, 20} is chosen per method on dev (nDCG@10 against the single training positive, full-corpus retrieval). Test relevance judgments are used only for the final scoring.

### 3.4 Exact fast scoring
With batch size 1, DyVo's query word weights and each entity weight do not depend on the other candidates: entity logits are computed per entity and max-pooled over tokens. Each query is therefore encoded once with every candidate it could receive. Any selected set S then scores as word(q,d) + Σ_{e∈S} s_e(q)·s_e(d) over the dumped index. This is the same as re-encoding the query with S, and makes the 5 × ~12 method/K grid cheap.

### 3.5 Comparisons (test, 42 CODEC queries, full corpus)
B0 DyVo word branch only · B1 REL · B2 GPT-4 / Mixtral (released) · B3 BM25 (released) · B4 LaQue (released) · C0 whole pool · controls that rank the same pool by DyVo weight, LaQue score, BM25 score, rarest-first (DF), random, REL-first · selectors E1-imit, E1-occ, E1-pair, **E1-loo**.
Statistics: paired randomization test and 95% bootstrap CI on per-query nDCG@10. Holm correction over the predeclared family: E1-loo vs B1 REL, E1-imit, E1-occ, E1-pair, the best dev control, and GPT-4.

## 4. Findings so far

### 4.1 Most BM25/LaQue candidates can never match a document
Share of the released test-query candidates that have at least one linked posting in CODEC:

| Source | Candidates (42 queries) | With ≥ 1 document posting |
|---|---:|---:|
| GPT-4 | 596 | 89.3% |
| Mixtral | 448 | 90.4% |
| REL | 196 | 80.6% |
| BM25 (authors) | 840 | 36.3% |
| LaQue (authors) | 840 | 23.8% |

Three quarters of the authors' LaQue candidates and nearly two thirds of their BM25 candidates have zero postings, so they cannot change any score under DyVo's exact-ID matching. LLM lists are much better grounded in the corpus. This supports the plan's hypothesis that candidate usefulness, not plausibility, matters.

### 4.2 DyVo's head cannot be used as a catalog-wide retriever
We tried using the frozen DyVo query head itself to score all 5.28M entities (and the 686k indexable ones). Its top entities are unrelated: for "How has the UK's Open Banking Regulation benefited challenger banks?" the top three are *The Rape of the Lock*, *Beating Retreat* and *The Celebrity*. The head was trained only to weight given candidates, and unnormalised max-over-token dot products favour hub embeddings. We use it only to weight pool members. Even within the pool, DyVo's weight ranks junk high (*Professor Challenger* 0.67 vs *Banking in the United Kingdom* 0.66), which is the gap a selector should close.

### 4.3 The dumped index is the reproduction
Scoring the 42 test queries with REL candidates on the dumped index (queries encoded at batch 64, as in the reproduction) gives nDCG@10 53.69 / nDCG@20 50.77 / R@1000 70.82, against 53.70 / 50.78 / 70.82 for the official 100k run. The 0.01 difference comes from storing weights in fp16. At batch size 1, used for every extension run, REL gives 53.67. A full-corpus test run now takes about 1 s instead of about 15 h.

### 4.4 Labels behave differently from every single feature
Over the 349,219 training (query, candidate) rows: `imit` is positive for 2.4% of candidates, `occ` 3.4%, `pair` 3.3% (7.1% negative), `loo` 3.3% (**7.4% harmful**, i.e. removing them would raise the teacher-weighted likelihood). Correlations between positive indicators: occ–pair 0.98, occ–loo 0.54, imit–loo 0.48. No raw feature predicts `loo` (DyVo weight −0.01, LaQue 0.005, BM25 0.02, REL membership 0.11), so the selector has to learn a combination.

### 4.5 The cheap pool reaches only a quarter of what GPT-4 proposes
Of GPT-4's 532 indexable test-query entities, only 125 (23.5%) are in our REL ∪ BM25 ∪ LaQue pool; Mixtral 20.7%. A selector cannot recover what is not in its pool (plan risk #1), which caps v1.

We added a fourth cheap source: corpus feedback (PRF) = the top-20 entities linked in the top-10 documents of DyVo's word-only first pass, weighted by document score × entity weight. It uses no relevance judgments and every candidate has postings by construction. Coverage of GPT-4's indexable entities, analysis only:

| PRF setting | PRF alone | PRF ∪ pool |
|---|---:|---:|
| top-5 docs, 20 ents | 26.1% | 39.8% |
| **top-10 docs, 20 ents (used)** | 28.4% | **41.5%** |
| top-20 docs, 40 ents | 35.7% | 47.9% |

We fixed 10 docs / 20 entities a priori (20 = the K of the other generators) rather than taking the best cell. This table was computed against GPT-4's test lists **before** choosing, so we report all cells (they differ by ≤ 8 points).

## 5. Assumptions

| # | Assumption | Why |
|---|---|---|
| E1-A1 | Frozen model = reproduced DyVo full100k (REL-trained, our run; nDCG@10 53.70) | the plan's "verified baseline"; no released trained checkpoint |
| E1-A2 | Teacher for the imitation control = REL linker lists | no LLM candidate lists exist for training queries. The released GPT-4/Mixtral lists cover only the 42 test queries and must not be used for training. Local Qwen3 models are cached, but the pinned transformers 4.40 cannot run Qwen3 and AWQ kernels are not installed. So this is **utility learning, not LLM distillation** (council report §9). |
| E1-A3 | Training positives/negatives = synthetic Mixtral training queries + MonoT5-3B scores shipped by the authors | the only training relevance signal for CODEC; test qrels are never used for labels, K, or features |
| E1-A4 | Query encoding at batch size 1 for every method, including baselines | removes the padding effect on max-pooling (reproduction assumption A7); baselines are recomputed under the same rule, so B1 may differ slightly from the 53.70 run at batch 64 |
| E1-A5 | Pool restricted to entities with ≥ 1 linked document | zero-posting entities cannot change scores under the fixed index; must be recomputed if the index changes (Extension 2) |
| E1-A6 | Dev metric = nDCG@10 against one synthetic positive per training query | only legitimate tuning signal; test topics are not tuned on |

## 6. Challenges

1. **No training-query LLM teacher.** The plan's teacher-distillation control needs LLM entity lists for training queries. Only test-query lists were released. We substituted REL imitation and record that we cannot claim to retain "GPT-level" benefit (E1-A2).
2. **DyVo head is not a retriever** (§4.2): the first "cheap dense" generator design failed and was replaced with the authors' LaQue encoder.
3. **Evaluation cost.** The official evaluator needs about 15 h per run. We dump the index once (~1.5 h) and use exact per-entity decomposition (§3.4), which makes each of the ~70 method/K test runs take seconds.
4. **Small test set.** 42 topics, so differences under about 1 nDCG@10 point are within noise; all claims use paired tests with Holm correction.
5. **Weak tuning signal.** The only legitimate dev data are synthetic training queries, each generated from its positive document. The word branch alone already reaches dev nDCG@10 0.829 (REL 0.834), and adding the whole pool *hurts* (0.801). Dev therefore picked K = 1 for every learned method, which compresses differences between selectors. We also report the full fixed-K grid on test (descriptive; nothing chosen on it) and a secondary test family at K = 5 and K = 20 (K = 20 is the council report's predeclared starting condition).
6. **Memory blow-ups and OOM kills.** Four runs were killed silently by the host OOM killer (exit 137; the session cgroup records `oom_kill 4`). Profiling showed our own bug: indexing an `NpzFile` re-reads the whole array on **every** access, and storing slices of it pinned one 26 MB copy per query, reaching 115 GB RSS. Fixed by materialising arrays once. We also store the index in a precomputed int32 CSC layout that is memory-mapped: load drops from > 20 GB peak / 60 s to 0.6 GB / 1 s.
7. **Shared server.** Another user's jobs share GPU0, so corpus-pass throughput varied between 9 and 211 docs/s. All long jobs are resumable by shard.

## 7. Results

### 7.1 v1: pool = REL ∪ BM25-title ∪ LaQue (dev-chosen K; test = 42 CODEC queries, full corpus, batch-1 query encoding)

| Run | K | nDCG@10 | nDCG@20 | R@1000 |
|---|---:|---:|---:|---:|
| B0 DyVo word branch only | 0 | 52.95 | 48.84 | 69.07 |
| B1 REL | native | 53.67 | 50.75 | 70.82 |
| B2 GPT-4 (released) | native | **56.88** | **53.61** | **74.81** |
| B2 Mixtral (released) | native | 55.12 | 52.62 | 73.39 |
| B3 BM25 (released) | 20 | 53.07 | 49.50 | 70.13 |
| B4 LaQue (released) | 20 | 53.40 | 50.16 | 71.32 |
| C0 whole pool (~40) | all | 53.82 | 50.14 | 72.29 |
| C ranked by DyVo weight | 1 | 52.56 | 48.90 | 69.02 |
| C ranked by LaQue score | 1 | 52.86 | 50.26 | 70.32 |
| C ranked by BM25 score | 1 | 53.94 | 50.89 | 70.66 |
| C rarest-first (DF) | 10 | 52.92 | 48.81 | 69.07 |
| C random | 1 | 52.74 | 49.06 | 69.32 |
| C REL-first (best dev control) | 1 | 52.78 | 49.85 | 69.85 |
| E1-imit (REL imitation) | 1 | 53.59 | 49.42 | 69.83 |
| E1-occ (document occurrence) | 1 | **54.15** | 50.46 | 70.35 |
| E1-pair (pairwise utility) | 1 | 54.02 | 50.53 | 70.58 |
| **E1-loo (listwise utility)** | 1 | 54.03 | 50.43 | 70.55 |

Fixed-K grid on test (nDCG@10; descriptive, nothing was selected on it):

| Method | K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---:|---:|---:|---:|---:|
| E1-imit | 53.59 | 54.01 | 52.94 | 53.55 | 52.93 |
| E1-occ | 54.15 | 53.59 | 54.47 | 53.34 | 53.73 |
| E1-pair | 54.02 | 53.61 | **54.62** | 53.47 | 53.79 |
| E1-loo | 54.03 | 54.00 | 54.25 | 53.29 | 53.90 |
| C DyVo weight | 52.56 | 52.50 | 52.76 | 53.30 | 53.33 |
| C LaQue score | 52.86 | 52.75 | 53.05 | 53.47 | 52.65 |
| C BM25 score | 53.94 | 54.00 | 54.15 | 54.36 | 54.05 |
| C random | 52.74 | 52.51 | 52.43 | 51.86 | 52.79 |
| C REL-first | 52.78 | 53.54 | 53.71 | 53.95 | 53.24 |

R@1000 at K=5: E1-loo 71.54, E1-pair 71.59, E1-occ 71.26, E1-imit 70.87, C DyVo weight 69.73, C random 69.59 (REL 70.82).

**Statistics** (paired randomization test, nDCG@10, Holm over each family):

| E1-loo vs | Dev-chosen K: Δ (p, p_Holm) | K = 5: Δ (p, p_Holm) | K = 20: Δ (p, p_Holm) |
|---|---|---|---|
| B1 REL | +0.36 (0.56, 1.0) | +0.59 (0.52, 1.0) | +0.23 (0.85, 1.0) |
| E1-imit | +0.44 (0.62, 1.0) | +1.31 (0.10, 0.80) | +0.97 (0.11, 0.80) |
| E1-occ | −0.13 (0.81, 1.0) | −0.21 (0.67, 1.0) | +0.16 (0.50, 1.0) |
| E1-pair | +0.01 (1.0, 1.0) | −0.37 (0.42, 1.0) | +0.10 (0.75, 1.0) |
| best simple control | +1.25 vs REL-first (0.06, 0.30) | +0.11 vs BM25 (0.92, 1.0) | −0.15 vs BM25 (0.90, 1.0) |
| C LaQue score | | +1.20 (0.21, 1.0) | +1.25 (**0.037**, 0.30) |
| B2 GPT-4 | −2.85 (**0.034**, 0.20) | | |

**v1 reading.**
1. **Document-derived supervision beats imitation and generic ranking of the same pool, but not significantly.** Every learned selector trained on a document signal (occ, pair, loo) sits 0.5–1.5 nDCG@10 above the DyVo-weight, LaQue, random and rarest-first controls at almost every K, and above REL imitation. No difference survives Holm correction on 42 topics.
2. **The listwise utility objective adds nothing over simple labels.** loo ≈ pair ≈ occ (|Δ| ≤ 0.4, p ≥ 0.4). By the plan's own falsifier ("if document-utility training does not beat the best simple label … the complex training claim fails"), we **do not** claim the leave-one-out objective is better. The defensible finding is the narrower one: *any* document-grounded label helps relative to imitation or generic scores.
3. **A cheap baseline is hard to beat.** Ranking the pool by BM25 title score is as good as the learned selectors (54.0–54.4).
4. **The pool, not the selector, is the bottleneck.** All selectors sit about 3 points under GPT-4, and the pool contains only 23.5% of GPT-4's indexable entities (§4.5). This motivated v2.
5. **DyVo's own entity weight is a poor selector.** Ranking by it is the worst non-random control (52.6 at K=1). DyVo's head was trained to weight REL candidates, not to choose among noisy ones.

### 7.2 v2: pool + corpus feedback (PRF)
Pool = REL ∪ BM25-title ∪ LaQue ∪ PRF (about 58 candidates per query; PRF adds 17.9 new ones per test query and 18.0 per training query). Selectors are retrained with 4 extra features: PRF score, PRF rank, PRF membership, PRF score relative to the pool maximum. Everything else is identical to v1 (`--pool _prf --tag _prf`; outputs `scratch/ext/e1/results_prf/`).

Dev again picked K = 1 for every learned method, and on dev the learned selectors got **worse** as K grew (E1-loo 0.820 → 0.716 from K=1 to K=20). On the synthetic dev queries, PRF entities come from documents that include the positive, so all labels reward them, but in full-corpus retrieval broad PRF entities (e.g. *Reuters*, *United States*) raise thousands of other documents as well. The labels are computed over each query's ~100 teacher-scored documents and cannot see that corpus-wide effect.

Test, dev-chosen K (all K = 1 except rarest-first):

| Run | nDCG@10 | nDCG@20 | R@1000 |
|---|---:|---:|---:|
| B1 REL | 53.67 | 50.75 | 70.82 |
| B2 GPT-4 / Mixtral (released) | 56.88 / 55.12 | 53.61 / 52.62 | 74.81 / 73.39 |
| C0 whole v2 pool (~58) | 50.97 | 48.74 | **75.24** |
| C ranked by PRF score | 53.28 | 50.24 | 70.28 |
| E1-occ / pair / loo / imit | 52.59 / 53.98 / 52.89 / 53.05 | | 70.06 / 70.17 / 70.07 / 69.70 |

Test fixed-K grid (descriptive):

| Method | K=1 | K=3 | K=5 | K=10 | K=20 | R@1000 at K=10 |
|---|---:|---:|---:|---:|---:|---:|
| E1-imit | 53.05 | 53.62 | 54.27 | 52.98 | 51.55 | 72.19 |
| **E1-occ** | 52.59 | **55.46** | 55.25 | 55.07 | 52.85 | **74.91** |
| E1-pair | 53.98 | 54.23 | 54.61 | 53.07 | 52.37 | 74.60 |
| E1-loo | 52.89 | 53.18 | 52.62 | 51.24 | 51.46 | 73.31 |
| C PRF score | 53.28 | 54.61 | **55.48** | 53.57 | 53.02 | 74.08 |
| C BM25 score | 53.94 | 54.00 | 54.15 | 54.36 | 54.05 | 71.00 |
| C DyVo weight | 52.48 | 52.15 | 51.87 | 51.00 | 51.44 | 70.78 |
| C random | 53.23 | 53.05 | 53.42 | 52.95 | 54.35 | 70.95 |

### 7.3 Choosing K without touching a topic's own judgments (nested topic-disjoint selection)
The dev proxy is unusable for K, as challenge 5 explains. Following the council report's small-topic discipline (§11), we choose K on a random half of the 42 test topics and evaluate on the other half, then swap. No topic is ever scored with a K chosen on itself. We repeat this over 50 random splits (`e1_cvk.py`; `results*/cvk.json`).

| Method (CV-chosen K) | v1 nDCG@10 | v1 R@1000 | v2 (+PRF) nDCG@10 | v2 R@1000 |
|---|---:|---:|---:|---:|
| B1 REL (fixed) | 53.67 | 70.82 | 53.67 | 70.82 |
| B2 Mixtral (fixed) | 55.12 | 73.39 | 55.12 | 73.39 |
| B2 GPT-4 (fixed) | 56.88 | 74.81 | 56.88 | 74.81 |
| C BM25 score | 53.75 ± 0.34 | 70.91 | 53.75 ± 0.34 | 70.91 |
| C PRF score | – | – | **54.75 ± 1.14** | **73.36** |
| C DyVo weight | 52.23 ± 0.52 | 70.63 | 51.49 ± 0.78 | 69.71 |
| C random | 52.08 ± 0.41 | 70.06 | 53.37 ± 0.82 | 71.83 |
| E1-imit | 53.18 ± 0.49 | 70.89 | 53.36 ± 0.91 | 71.23 |
| **E1-occ** | 53.52 ± 0.50 | 71.22 | **54.63 ± 0.65** | **73.34** |
| E1-pair | 53.60 ± 0.57 | 71.41 | 53.52 ± 1.01 | 72.60 |
| E1-loo | 53.34 ± 0.42 | 71.38 | 51.81 ± 1.00 | 71.05 |

(± = standard deviation over the 50 splits.) Paired tests on split 0, Holm within each family:
- v2 E1-occ vs REL: +0.88 (p = 0.49); better than REL in **94% of the 50 splits**. vs Mixtral −0.57 (p = 0.69); vs GPT-4 −2.33 (p = 0.18); vs PRF-score ranking −0.93 (p = 0.24).
- v2 **E1-loo is significantly worse** than PRF-score ranking (−3.63, p_Holm = 0.025) and than REL imitation (−2.42, p_Holm = 0.044); it beats REL in 0% of splits.
- v1: no learned selector differs from REL or from the BM25 control (all p ≥ 0.25); only GPT-4 is significantly better than E1-loo (p_Holm = 0.016).

### 7.4 Latency (`e1_latency.py`; uncached, batch 1, 42 queries × 3 repeats, random order, shared A5000)

| Stage (p50 ms) | REL replay | E1 v1 | E1 v2 (+PRF) |
|---|---:|---:|---:|
| BM25-title search | – | 49.2 | 20.9 |
| LaQue encode + search (686k entities) | – | 10.3 | 7.6 |
| PRF word-only first pass | – | – | 271.1 |
| DyVo weights for pool + features + selector | – | 15.6 | 11.1 |
| DyVo query encode | ~300 | 305.8 | 257.9 |
| Full-corpus scoring (scipy, CPU) | ~20 | 19.9 | 12.2 |
| **Total p50 (p95)** | **322.8 (359.9)** / 271.4 (317.1) | **407.0 (448.4)** | **589.1 (633.0)** |

Peak GPU memory 1.76 GB (LaQue indexable table in fp16 + both encoders). REL linking itself and live GPT-4/Mixtral generation **cannot be timed**: only their cached outputs exist (replay). So we cannot claim a measured speed-up over an LLM. E1 v1 adds +84 ms (+26%) to the REL replay. v2 adds +318 ms, mostly because our PRF pass re-encodes the query for the word-only run; one shared forward pass would remove most of that, but we did not implement it.

### 7.5 E1 + E2: selectors recomputed on the repaired document index (plan step 8)
Same pipeline as v2 (§7.2–7.3), but the index is the E2 v2 grounded c=5 index (`scratch/ext/index_grounded_v2`). The PRF pool, document-occurrence and utility labels, selectors and nested CV-K are all recomputed on that index (`run_after_delta.sh`; `results_prf_g2/`). Fixed query sides on the repaired index: REL 54.14, GPT-4 56.26, Mixtral 54.73 (vs 53.67 / 56.88 / 55.12 on the linked index).

| Method (CV-chosen K, 50 splits) | linked index nDCG@10 | R@1000 | repaired index nDCG@10 | R@1000 |
|---|---:|---:|---:|---:|
| C BM25 score | 53.75 ± 0.34 | 70.91 | 53.49 ± 0.22 | 71.27 |
| C PRF score | **54.75 ± 1.14** | 73.36 | 53.34 ± 0.94 | 73.59 |
| C DyVo weight | 51.49 ± 0.78 | 69.71 | 51.38 ± 0.34 | 70.52 |
| C random | 53.37 ± 0.82 | 71.83 | **54.74 ± 0.69** | 71.73 |
| E1-imit | 53.36 ± 0.91 | 71.23 | 53.17 ± 0.74 | 71.40 |
| **E1-occ** | 54.63 ± 0.65 | 73.34 | 54.01 ± 0.99 | **73.89** |
| E1-pair | 53.52 ± 1.01 | 72.60 | 53.27 ± 0.80 | 73.14 |
| E1-loo | 51.81 ± 1.00 | 71.05 | 50.79 ± 0.70 | 72.37 |

Paired tests on split 0 (repaired index, Holm within each method's family):
- E1-occ vs REL −1.05 (p = 0.45); better than REL in only **54%** of splits (94% on the linked index). vs GPT-4 −3.16 (p = 0.054, Holm 0.65); vs PRF-score +1.08 (p = 0.084).
- E1-loo is again the worst selector. It is significantly worse than GPT-4 (−5.94, p_Holm = 0.016), Mixtral (−4.41, p_Holm = 0.021) and random selection (−4.95, p_Holm = 0.007), and beats REL in 0% of splits.
- With only 42 topics, the random control comes out high (54.74). It is not significantly better than E1-occ (+2.17, p = 0.095), which shows how noisy differences of this size are.

**Reading.** The extensions do not combine well. On repaired documents every learned selector and the PRF ranking lose 0.2–1.4 nDCG@10 points. Only R@1000 rises (+0.2 to +1.3), as in E2. The repaired index also raises REL by 0.5, so E1-occ's margin over REL disappears. This matches E2's finding that repair adds matches with no ranking signal: corpus-feedback candidates, now drawn from noisier postings, become less precise.

## 8. Conclusions for Extension 1

1. **RQ1 (does utility supervision beat generic ranking and imitation at the same pool and K?): not supported.** With topic-disjoint K selection, no learned selector is significantly better than REL, BM25-title ranking or simple PRF-score ranking. The leave-one-out listwise utility label, the plan's main E1 objective, is the **worst** label in v2 and significantly worse than plain PRF ranking and REL imitation. By the plan's falsifier, the complex-objective claim is rejected.
2. **What does help is grounding the candidate pool in the corpus.** Adding corpus-feedback (PRF) candidates and filtering them with a simple document-occurrence selector (E1-occ) gives nDCG@10 54.6 / R@1000 73.3. That is better than REL in 94% of topic splits and statistically indistinguishable from Mixtral (55.1 / 73.4), with no LLM at query time. Simply ranking PRF candidates by their feedback score does as well (54.8 / 73.4). The useful ingredient is the *source* (entities co-occurring with the query's best lexical matches), not the learned objective.
3. **Most entity candidates are useless because the documents don't contain them.** 76% of the authors' LaQue and 64% of BM25 test candidates have no posting anywhere in CODEC (vs 11% for GPT-4). DyVo's exact-ID matching makes "is this entity in the index?" the first-order question.
4. **Cost:** a v1-style selector adds about 84 ms to a ~320 ms query; v2 adds about 320 ms in our unoptimised implementation. Both avoid any LLM call, but we cannot quantify the saving against a live LLM.
5. **Combining with Extension 2 does not help** (§7.5): on the evidence-repaired index every learned selector and the PRF ranking lose 0.2–1.4 nDCG@10 points (only R@1000 rises), and E1-occ no longer beats REL.
6. **Limitations:** single seed; 42 test topics; the REL linker is the only imitation teacher available for training queries; the dev proxy (synthetic queries) cannot tune K; the PRF setting was fixed after seeing a coverage table computed on GPT-4's test lists (§4.5).

## 9. Follow-up (25 Nov): comparison with baseline DyVo
See `result.md`. With K and α chosen by topic-disjoint CV, both extensions together reach nDCG@10 55.03 (+1.36 over DyVo, not significant) and R@1000 73.32 (+2.99, p_Holm = 0.002). The recall gain comes from E1 (E1 alone: +2.88, p_Holm = 0.002).
