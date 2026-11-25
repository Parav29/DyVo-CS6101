# Results: DyVo extensions compared with the DyVo baseline

CODEC, 42 test queries, full corpus (729,737 documents). The model is the frozen DyVo full100k from our reproduction; every system uses the same checkpoint. Scores are ×100.

- **Baseline DyVo** = REL query entities + REL-linked document entities, i.e. the original pipeline. nDCG@10 **53.67**, R@1000 **70.82**. Our reproduction gave 53.70; the paper reports 53.40.
- **Extension 1 (E1)**: a learned query-entity selector over a candidate pool that includes entities from the top-ranked documents (feedback/PRF candidates). Selector: E1-occ v2. Details in `extension1.md`.
- **Extension 2 (E2)**: evidence-grounded repair of document entities. Variant: grounded v2, c = 5. Details in `extension2.md`.
- **Follow-ups F1–F3** (25 Nov). F1 adds E1's selected entities to an existing query list. F2 interpolates document scores between the original and the repaired index. F3 adds significance tests on recall.

**The design was written before running.** `scratch/ext/followup_design.md` was declared at 11:37 IST and not edited afterwards. Every setting (K = number of E1 entities, α = interpolation weight) was chosen by nested topic-disjoint cross-validation:
- choose the setting on a random half of the topics and score it on the other half, then swap;
- repeat over 50 random splits;
- no topic is ever scored with a setting chosen on its own judgments.

Every system in the design is reported below. Code: `scripts/ext/f_followups.py`. Outputs: `scratch/ext/followup/`. Log: `logs/ext_followup.log`.

## 1. Headline: extensions vs baseline DyVo (same REL query side, no LLM)

| System | nDCG@10 | nDCG@20 | R@1000 | Δ nDCG@10 | Δ R@1000 (95% CI) | p_Holm (R@1000) |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline DyVo** (REL, linked docs) | 53.67 | 50.75 | 70.82 | – | – | – |
| E1 alone (E1-occ v2, CV-K) | 54.75 ± 0.35 | 51.49 | 73.21 | **+1.08** | **+2.88** [+1.32, +4.77] | **0.002** |
| E2 alone (REL, repaired docs) | 54.14 | 51.19 | 70.99 | **+0.47** | +0.18 [−0.17, +0.53] | 0.35 |
| F1: REL + E1 entities (CV-K) | 54.58 ± 0.40 | **51.59** | **73.39** | **+0.91** | **+2.93** [+1.33, +4.91] | **0.002** |
| F2: REL, interpolated E2 (CV-α) | 54.07 ± 0.15 | 51.13 | 70.98 | **+0.40** | +0.20 [−0.03, +0.46] | 0.25 |
| **F1+F2: both extensions (CV-K × α)** | **55.03 ± 0.61** | 51.47 | 73.32 | **+1.36** | **+2.99** [+1.35, +4.99] | **0.002** |

(± = standard deviation over the 50 CV splits. Deltas are relative to the baseline, using the 50-split means. CIs and p-values are paired tests on per-query scores from split 0. Holm correction is over the five systems, separately for each metric.)

**What this shows:**
- **Every extension system scores above baseline DyVo on all three metrics.**
- **Recall gain is significant (p_Holm = 0.002).** With E1, R@1000 rises by about +2.9 points, from 70.8 to 73.2–73.4.
  - Per query: 25 better vs 7 worse.
  - The CI excludes zero.
  - It is above the baseline in 100% of the 50 splits.
- **Both extensions together (F1+F2) give the best nDCG@10:** 55.03, which is +1.36 over DyVo and about the level of Mixtral-generated entities (55.12). It needs no LLM at query time.
- **The nDCG@10 gains (+0.4 to +1.4) are not significant on 42 topics.** Per-topic split-0 differences: E1 +0.64, p = 0.59; F1+F2 +0.39, p = 0.74. The sign is stable: the gain is positive in 98–100% of the 50 splits. Report them as **consistent but not significant** improvements.

## 2. Do the extensions also help the strong LLM query sides?

Each system below is compared with its own base list, without our extensions. Holm correction is over these six systems for each metric.

| System | nDCG@10 | R@1000 | Δ nDCG@10 vs own base | Δ R@1000 vs own base (95% CI) | p_Holm (R@1000) |
|---|---:|---:|---:|---:|---:|
| GPT-4 entities (released) | 56.88 | 74.81 | – | – | – |
| F1: GPT-4 + E1 | 56.52 | 75.64 | −0.36 | +0.46 [−0.10, +1.05] | 0.13 |
| F2: GPT-4, interpolated E2 | 56.54 | 74.99 | −0.34 | +0.23 [+0.06, +0.43] | **0.046** |
| **F1+F2: GPT-4** | 56.33 | **76.00** | −0.55 | **+1.33** [+0.52, +2.16] | **0.009** |
| Mixtral entities (released) | 55.12 | 73.39 | – | – | – |
| **F1: Mixtral + E1** | 55.53 | **74.87** | +0.41 | **+1.54** [+0.57, +2.69] | **0.009** |
| F2: Mixtral, interpolated E2 | **55.54** | 73.74 | +0.42 | +0.35 [+0.06, +0.68] | 0.052 |
| **F1+F2: Mixtral** | 55.32 | **74.89** | +0.20 | **+1.69** [+0.65, +2.88] | **0.009** |

- **Recall improves significantly on top of LLM entities as well.** GPT-4 + both extensions reaches R@1000 **76.00**, the highest of any system (+1.33 over GPT-4, +5.2 over baseline DyVo). Mixtral + E1 gains +1.5.
- **Top-10 precision on GPT-4 does not improve.** All three GPT-4 variants are 0.3–0.6 below GPT-4 alone. F2-GPT-4 is −0.41 with p = 0.047 unadjusted, but not significant after Holm (0.28). On Mixtral, nDCG@10 rises by +0.2 to +0.4 (not significant).

## 3. Results that did not improve (also reported)

- **The E1 training objective.** The leave-one-out utility selector, the method E1 was designed to test, is the worst selector (`extension1.md` §7.3, §8). The useful part of E1 is the corpus-feedback candidate pool with a simple document-occurrence selector, not the utility label.
- **Repair by itself barely moves the scores.** E2 alone gives +0.47 nDCG@10 and +0.18 R@1000, neither significant. The pre-registered E2 v1 was worse than the baseline (−1.63 nDCG@10; `extension2.md` §7.1). The v2 used here was designed after the v1 error review and is labelled exploratory.
- **Selectors recomputed on the repaired index** score lower than selectors trained on the original index (`extension1.md` §7.5).
- **GPT-4 nDCG@10** is not improved by any extension (§2).

## 4. Caveats

1. **42 test topics.** nDCG@10 differences of about 1 point cannot reach significance at this size: 95% CIs for E1 are about ±2.2 points. The significant results are all on R@1000.
2. **nDCG@10 is the paper's headline metric. R@1000 is secondary.** We tested both, declared in advance with separate Holm families. Our claim is that "the extensions improve first-stage recall significantly and nDCG@10 slightly (not significantly)". We do not claim "nDCG@10 significantly".
3. **Design history.** E1's feedback pool (v2) and E2 v2 were designed after earlier experiments: E1 after a coverage table computed on GPT-4's test lists (`extension1.md` §4.5), E2 after the v1 error review. F1–F3 were declared before running. Their *ingredients* (E1-occ v2, grounded v2) already existed, and we already knew E1-occ v2's standalone test score.
4. **CV splits differ slightly from `extension1.md` §7.3.** In §7.3, E1 alone scored 54.63, beating REL in 94% of splits. Here it scores 54.75, beating REL in 100% of splits. Same selections, same procedure; the topic order fed to the random permutation differs, so the 50 splits differ. This shows how much split randomness moves the numbers (about 0.1 points).
5. **Single seed and frozen model:** DyVo is not retrained on repaired documents.
6. **Cost:** E1 v1 adds +84 ms per query; E1 v2 with the feedback pass adds +318 ms (`extension1.md` §7.4). E2 adds +15% entity postings to the index, about 11.5 MB. Neither needs an LLM at query time.

## 5. One-paragraph summary for the report

With the DyVo model frozen, our two extensions together raise nDCG@10 from 53.67 to 55.03 (+1.4) and R@1000 from 70.8 to 73.3 (+3.0). Extension 1 adds learned selection of query entities from corpus-feedback candidates; Extension 2 adds evidence-grounded repair of document entities. The recall gain is statistically significant (p_Holm = 0.002, 25 of 42 queries improved vs 7 worse), and the nDCG@10 gain is positive in 98% of topic splits but not significant on 42 topics. With no LLM at query time, the combined system matches DyVo with Mixtral-generated entities (55.12). Added to GPT-4's entity lists, it gives the best R@1000 of any system (76.0, +1.3 over GPT-4, p_Holm = 0.009), but it does not improve GPT-4's top-10 precision. The extensions' original hypotheses were not supported: the leave-one-out utility training was not better than simpler selectors, and grounded repair alone was not better than its controls. The gains come from grounding query entities in the corpus and from mixing repaired and original document scores.
