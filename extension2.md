# Extension 2: evidence-grounded document entity repair

**Status:** complete (23–25 Nov 2026). v1 = pre-registered design, v2 = exploratory follow-up; conclusions in §8.
Plan source: `EXTENSIONS.md` §Extension 2. (The council report and detailed plan cited below were internal planning notes, not included here.)

## 1. What we are testing

An entity helps DyVo only if the query and a relevant document carry the **same entity ID**. DyVo's documents use REL-linked entities only, so a useful query entity is wasted when linking missed it in a relevant document. We add a small number of extra document entities, **only where the document text explicitly mentions them**. Every variant gets the same posting budget, so no variant wins just by storing more.

```text
Original:  document text -> REL-linked IDs -> DyVo doc entity head -> entity postings
Ours:      same text -> linked + proposals (LaQue dense, title/alias mentions)
                     -> keep only proposals with an explicit mention in the visible 512-token window
                     -> rank by LaQue doc-entity similarity, per-doc cap + global budget
                     -> same DyVo doc entity head -> rebuilt entity postings (separate index)
```

Unchanged: word encoder and word postings, query encoder, DyVo scoring, frozen model (DyVo full100k).

**Research question (RQ2):** at a fixed entity-posting budget, does evidence-grounded repair improve retrieval over linked-only documents, alias-only repair and unfiltered expansion?

## 2. Progress log

| Date | Step | Status |
|---|---|---|
| 23 Nov | Shared infrastructure with Extension 1 (`scripts/ext/common.py`, `dump_index.py`) | done |
| 23 Nov | Wikipedia redirect + page dumps (Wikimedia, `enwiki-latest-{redirect,page}.sql.gz`) for the alias table | done |
| | `e2_alias.py`: lexicon of titles, de-qualified titles and redirects → catalog entity | written |
| | `e2_mentions.py`: visible-window mention detection for all 729,737 docs | written |
| | `e2_dense.py`: LaQue document proposals (top-30, full catalog) + doc embeddings | written |
| | `e2_build.py`: alias-only / unfiltered / grounded variants with a matched budget; union candidate file | written |
| 23 Nov | Mentions (20.9M pairs), LaQue proposals, alias lexicon (12.5M forms) | done |
| 23 Nov | Headroom gate (§4.2) | passed |
| 23 Nov | Variant build (budget B = 3.58M additions, +28% over linked) | done |
| 23 Nov | Union dump (one corpus pass, 22.16M entity postings) | done; linked postings identical to baseline |
| 23 Nov | `e2_run.py`: 13 document variants × 5 query sides, diagnostics, storage, statistics | **done: v1 results §7.1** |
| 23 Nov | Blinded review of 45 added entities | done: 38% / 38% / 15% correct (§7.2) |
| 23 Nov | **Exploratory v2** (stricter evidence, keyphraseness, CSLS ranking; designed after §7.2): delta dump + evaluation | done (§7.3) |
| 25 Nov | Session cut off by a server outage after the 23 Nov runs finished; v2 blinded review, v2 write-up and conclusions completed | done (§7.3, §8) |
| 25 Nov | Follow-ups F1 (base list + E1 entities), F2 (linked/repaired score interpolation), F3 (R@1000 tests); design declared first (`scratch/ext/followup_design.md`), `scripts/ext/f_followups.py` | done: `result.md` |
| | Local-evidence-span scoring ablation | not done (see §6) |

## 3. Method details

### 3.1 Visible text
DyVo encodes `title + " " + text` truncated to 512 word-pieces (`lsr/utils/dataset_utils.read_collection`, collator `d_max_length = 512`). Every mention we use as evidence must lie inside that window. We compute it with the same DistilBERT tokenizer and its offset mapping, so no variant sees extra text (plan requirement: same visible text for all variants).

### 3.2 Alias lexicon (`e2_alias.py`)
Surface forms map to LaQue catalog entities from three sources: the exact title, the title without a "(…)" qualifier, and every English Wikipedia redirect. Ambiguous forms resolve to the most popular entity, i.e. the smallest catalog index; `entity_list.json` starts United States, Race and ethnicity in the US Census, Association football, World War II…, a popularity order. List, index and disambiguation pages are excluded.

### 3.3 Mentions (`e2_mentions.py`)
Greedy longest match (up to 6 tokens) over the visible text. Single-token matches must be capitalised in the document (proper-noun evidence); multi-token matches may be any case. For each document we record entity, first character offset, count, source and surface string. The offsets allow a later local-span ablation and blinded review.

### 3.4 Variants (`e2_build.py`)
All additions exclude already-linked entities. Each variant gets per-doc cap c = 5 and the **same global number of added candidates B** (the smallest variant's available total). The budget is spread evenly across documents: rank within the document first, then score.

| Variant | Proposal source | Evidence check | Ranking | Plan row |
|---|---|---|---|---|
| linked | REL links | n/a | n/a | B1 |
| alias | title/alias mentions | mention by construction | mention count, then position | E2-C1 alias-only |
| unfiltered | LaQue doc top-30 | none | LaQue similarity | E2-C2 unfiltered expansion |
| **grounded** | LaQue top-30 ∪ mentions | **explicit mention required** | LaQue doc-entity similarity | **E2** |

### 3.5 One corpus pass for all variants
A document entity weight s_e(d) depends only on the document's tokens and on e: per-entity logits, max-pooled over tokens. So we encode the corpus once with linked ∪ all variants' additions, then build each variant's index by keeping word postings, linked postings and that variant's additions. Sanity check: the "linked" index rebuilt this way must equal the separately dumped baseline index.

### 3.6 Evaluation (`e2_run.py`)
2 × 2 (and more): query side ∈ {REL, GPT-4 released, E1 selector} × document side ∈ {linked, alias, unfiltered, grounded}. Metrics: nDCG@10 (primary), nDCG@20, R@1000. Statistics: paired randomization test with Holm correction over grounded vs {linked, alias, unfiltered} for each query side. Mechanism diagnostics: positive entity matches per judged-relevant vs judged-non-relevant document. These use test relevance judgments for analysis only; nothing is tuned on them. Storage: total and added entity postings, index bytes. Latency: scoring time p50/p95.

## 4. Findings so far

### 4.1 Evidence resources built
- **Lexicon:** 12,474,910 surface forms → 5,086,096 of the 5,277,580 catalog entities (96%). Sources: 5,014,170 exact titles, 195,162 de-qualified titles, 7,265,578 Wikipedia redirects (8.8M redirects resolved to catalog entities), no hash collisions.
- **Mentions:** 20,935,353 (document, entity) explicit-mention pairs in the visible windows of 729,737 documents, 28.7 per document. CPU, 24 workers, 3 min.
- **Dense proposals:** LaQue top-30 over the full catalog for every document plus document embeddings. GPU, 35 min.

Precision fixes found by inspection, before any retrieval run: the first mention pass accepted lowercase multi-word hits such as "on China", "and more", "it was", "the benefit" and "pull back", which match song, album or film titles. We added three rules: (a) a span may not start or end with a function word; (b) all-lowercase spans count only as multi-word exact titles, not redirects; (c) two-letter forms only when written all-uppercase ("UK", "EU"). On the example article, matches are now Apple → Apple Inc., Tim Cook, trade war, China, UK → United Kingdom, EU → European Union, Open banking.

### 4.2 Headroom gate (plan step 3): passed
Status of (query entity, document) pairs: linked = DyVo can match today; missing = not linked but explicitly mentioned in the visible text (what grounded repair can recover); absent = neither. `scratch/ext/e2/diag.json`.

| Query entities vs documents | Pairs | Linked | Missing but mentioned | Absent |
|---|---:|---:|---:|---:|
| Train: REL vs training positive | 8,496 | 90.7% | 2.4% | 6.8% |
| Train: LaQue top-5 vs training positive | 45,000 | 9.2% | 5.4% | 85.3% |
| Test: REL vs relevant | 17,768 | 12.5% | 5.4% | 82.1% |
| Test: REL vs non-relevant | 10,483 | 10.3% | 3.7% | 86.0% |
| Test: GPT-4 vs relevant | 54,431 | 13.8% | 4.6% | 81.6% |
| Test: GPT-4 vs non-relevant | 32,062 | 8.4% | 2.4% | 89.2% |
| Test: Mixtral vs relevant | 40,960 | 14.6% | 5.2% | 80.2% |
| Test: Mixtral vs non-relevant | 24,038 | 10.1% | 3.0% | 86.8% |

Reading (test rows are analysis only; nothing was tuned on them):
- Repairable headroom is real but modest. For GPT-4 candidates, recovering every "missing but mentioned" pair would add about a third more entity matches with relevant documents (4.6 on top of 13.8 points).
- The missing pairs are **more skewed toward relevant documents** than existing links: relevant/non-relevant ratio 1.9 for missing vs 1.65 for linked (GPT-4). Repair therefore adds discriminative matches, not just more matches.
- On training data, REL query entities are already 90.7% linked in the positive. Training queries were generated from those documents and linked by the same REL, so the training split shows little headroom for REL. The LaQue-pool entities show more (5.4% missing vs 9.2% linked).

## 5. Assumptions

| # | Assumption | Why |
|---|---|---|
| E2-A1 | Frozen model = reproduced DyVo full100k, not retrained on repaired documents | the plan's first experiment changes only the document candidates; retraining would confound the comparison |
| E2-A2 | Wikipedia redirects (current dump) are a valid alias source for the 2019/2020-era LaQue catalog | titles are matched exactly, so redirects to renamed pages are simply dropped |
| E2-A3 | "Most popular" sense = smallest catalog index | no link-probability (p(e|m)) table was released; REL's tables are several GB |
| E2-A4 | Capitalisation as evidence for single-token mentions | cheap precision filter; the plan forbids generator confidence as evidence but allows explicit mentions/aliases |
| E2-A5 | Budget = equal number of added (doc, entity) candidates; realised non-zero postings and bytes are reported separately | the DyVo head can give an added candidate weight 0 (ReLU), so equal candidates ≠ equal postings (council report §9B) |

## 6. Challenges

1. **No alias resource in the release.** `entity_list.json` has titles only, so we built an alias table from Wikipedia redirect dumps (2.6 GB download). The current dump format has one row per line, not many rows per `INSERT` line; our first parser found 0 redirects until this was fixed.
2. **Re-encoding cost.** A full corpus encode takes about 1.2–1.5 h on a shared A5000. The union-dump trick (§3.5) gives all 13 variants from one pass. v2 needed only a delta pass over the 525k documents with new pairs.
3. **String matches are not evidence.** The biggest scientific challenge (§7.2). Capitalised sentence starters, reporter bylines, page boilerplate ("5 Min Read", "File photo") and ambiguous names resolved to the wrong sense (Lowe's → *Lowe (band)*) produced most v1 additions. LaQue similarity made it worse by favouring "hub" entities that are close to every document (*Last Year*, *However (song)*).
4. **Shared machine / memory.** See extension1.md, challenge 6. `e2_build.py` initially took more than 25 min because indexing an `NpzFile` re-reads the archive each time (fixed: 2 min). A column-count bug crashed it after saving the variants; the saved files were verified intact.
5. **Local-span scoring ablation not run.** It needs a modified DyVo document forward (entity logits masked to the evidence span) and another corpus pass. Since repair did not help even with whole-document scoring, we prioritised the evidence-precision study (v2) instead.

## 7. Results

### 7.1 v1 (pre-registered design): 42 CODEC test queries, full corpus, frozen DyVo full100k
Budget curve: each variant adds up to c candidates per document (c ∈ {1, 2, 3, 5}). At c = 5 all three variants add exactly B = 3,575,930 candidates. Realised positive postings: alias 3,509,071, grounded similar, unfiltered (all candidates receive weight > 0 in > 98% of cases).

nDCG@10 (query side × document variant):

| Document variant | REL | GPT-4 | Mixtral | E1-occ v2 (K=3) | PRF score (K=5) |
|---|---:|---:|---:|---:|---:|
| linked (DyVo) | **53.67** | 56.88 | 55.12 | **55.46** | **55.48** |
| alias c=1 | **53.86** | 56.68 | 55.24 | 55.42 | 55.49 |
| alias c=2 | 53.56 | 57.11 | 55.08 | 55.11 | 55.48 |
| alias c=5 | 53.14 | **57.48** | 55.25 | 55.22 | 55.23 |
| unfiltered c=1 | 53.63 | 56.95 | **55.44** | 55.29 | 55.22 |
| unfiltered c=5 | 53.06 | 56.41 | 54.83 | 55.18 | 54.85 |
| grounded c=1 | 52.80 | 56.61 | 54.58 | **55.49** | 55.34 |
| grounded c=2 | 52.76 | 56.48 | 53.58 | 55.34 | 55.13 |
| grounded c=5 | 52.03 | 56.07 | 54.24 | 54.65 | 54.91 |

R@1000:

| Document variant | REL | GPT-4 | Mixtral | E1-occ v2 | PRF score |
|---|---:|---:|---:|---:|---:|
| linked | 70.82 | 74.81 | 73.39 | 72.19 | 73.46 |
| alias c=5 | 70.87 | 75.44 | **73.91** | **72.37** | 73.47 |
| unfiltered c=5 | 70.82 | 75.09 | 73.55 | 72.21 | 73.46 |
| grounded c=5 | 70.90 | **75.46** | 73.79 | 72.27 | **73.56** |

Statistics, grounded (c=5) vs controls, paired randomization, Holm within each query side:

| Query side | vs linked | vs alias c=5 | vs unfiltered c=5 |
|---|---|---|---|
| REL | −1.63 (p 0.026, Holm 0.078) | −1.11 (0.084, 0.17) | −1.03 (0.17, 0.17) |
| GPT-4 | −0.80 (0.43, 0.87) | −1.41 (**0.040**, 0.12) | −0.34 (0.68, 0.87) |
| Mixtral | −0.88 (0.37, 0.74) | −1.01 (0.15, 0.44) | −0.59 (0.50, 0.74) |
| E1-occ v2 | −0.81 (0.15, 0.30) | −0.57 (**0.028**, 0.083) | −0.53 (0.32, 0.32) |
| PRF score | −0.57 (0.23, 0.46) | −0.32 (**0.049**, 0.15) | +0.06 (0.91, 0.91) |

Mechanism diagnostics (positive entity-ID matches per judged document; analysis only):

| | linked | alias c=5 | unfiltered c=5 | grounded c=5 |
|---|---|---|---|---|
| REL: relevant / non-relevant | 0.580 / 0.461 (×1.26) | 0.723 / 0.536 (×1.35) | 0.597 / 0.467 (×1.28) | 0.715 / 0.527 (×1.36) |
| GPT-4: relevant / non-relevant | 1.957 / 1.142 (×1.71) | 2.250 / 1.281 (×1.76) | 2.014 / 1.159 (×1.74) | 2.290 / 1.289 (×1.78) |

Unsupported-addition rate (added entity not explicitly mentioned in the visible text): alias 0%, grounded 0% (by construction), **unfiltered 95.0%**.

**v1 reading.**
- Repair works as intended **mechanically**: grounded c=5 adds 23% more entity matches with relevant documents for REL and 17% for GPT-4, and the relevant:non-relevant ratio even improves slightly (1.26 → 1.36; 1.71 → 1.78).
- But it **does not improve ranking**. nDCG@10 falls for every query side, steadily with budget for REL and Mixtral. Only R@1000 rises a little for the LLM query sides (GPT-4 +0.65, Mixtral +0.40).
- **RQ2 as stated fails the falsifier.** Grounded repair is not better than linked-only documents, and it is *worse* than alias-only repair at the same budget (GPT-4 −1.41, p = 0.04; E1 side −0.57, p = 0.03; neither survives Holm).
- **Alias-only repair** is the most useful document variant for the strongest query side (GPT-4 57.48, +0.60; R@1000 +0.63), but it hurts REL at larger budgets.

### 7.2 Why: blinded review of what was added
We sampled 40 random additions per variant (sample files not included) and judged the first 45 shuffled items with the variant hidden; judgments are in `results/review_judgements.json`. A judgment counts as correct if the entity is the right sense and is supported or clearly relevant. The judge is Claude itself, blinded but **not independent**: one pass, no second rater.

| Variant | Judged correct | Precision |
|---|---:|---:|
| alias | 6 / 16 | 38% |
| grounded | 6 / 16 | 38% |
| unfiltered | 2 / 13 | 15% |

Most frequent additions over the whole budget:
- alias: *Min (god)* 96k and *Read (surname)* 76k (both from "5 Min Read" page headers), *Thursday (band)*, *File (tool)* ("File photo"), *Photograph*;
- grounded: *Standards (Jarrett album)* 64k, *Last Year* 51k, *However (song)* 45k, *First Time (TV series)*, *Despite (band)*;
- unfiltered: bilateral-relations pages (*Iceland–India relations* 8k).

Errors seen in the sample: wrong sense of an ambiguous surface (Lowe's → *Lowe (band)*, American Airlines → *American (word)*, NBA player → *David Thompson (explorer)*), reporter bylines (*Wee (surname)*, *Kang (Korean surname)*) and boilerplate. With roughly 60% wrong additions, extra matches mostly add noise for whichever query entity happens to share them. LaQue similarity, used for "salience", is attracted to hub entities, so ranking by it chose worse additions than plain mention counts.

### 7.3 Exploratory v2: tighter evidence (designed after §7.1–7.2; labelled post-hoc)
Three changes, each justified by the review rather than by test metrics:
1. **Single-token evidence:** only exact, unqualified titles, never sentence-initial (all-caps acronyms such as UK/EU allowed). This removes *Lowe (band)*, *However (song)*, *Photograph*, bylines. `E2_V2=1 e2_mentions.py` → 14.4M mention pairs (19.7/doc vs 28.7).
2. **Keyphraseness filter** (Mihalcea & Csomai 2007): add only entities that REL links in ≥ 10% of the CODEC documents mentioning them (linked_df / (linked_df + unlinked-mention df) ≥ 0.1, fixed a priori; 671,743 eligible entities). This removes weekdays, *Last Year*, *File (tool)*. Added entities are then occurrences the linker missed of entities it does link in this corpus.
3. **Hubness-corrected salience** for grounded ranking (CSLS-style): LaQue sim(d, e) − sim(mean document, e).

After the change, the most frequent additions are *Americans*, *Getty Images*, *Washington (state)* (still a wrong sense for "Washington", often meaning D.C.), *Presidency of Donald Trump*, *Federal government of the United States*. Budget B = 1,940,031 additions (+15%). The 1,282,339 new (doc, entity) pairs not in the v1 union are encoded in a delta pass over 525,199 documents (`--only-listed`).

**v2 nDCG@10** (`results_v2/test.json`; same frozen model, query sides and statistics as §7.1):

| Document variant | REL | GPT-4 | Mixtral | E1-occ v2 (K=3) | PRF score (K=5) |
|---|---:|---:|---:|---:|---:|
| linked (DyVo) | 53.67 | 56.88 | 55.12 | **55.46** | **55.48** |
| alias c=1 | **54.29** | 56.55 | 55.21 | 55.31 | 55.39 |
| alias c=2 | 54.08 | 56.63 | 55.06 | 55.30 | 54.93 |
| alias c=5 | 54.07 | 56.08 | 54.33 | 55.27 | 54.97 |
| unfiltered c=1 | 53.63 | **56.95** | **55.44** | 55.29 | 55.22 |
| unfiltered c=5 | 53.07 | 56.68 | 54.99 | 55.05 | 54.73 |
| grounded c=1 | 53.97 | 55.76 | 55.30 | 55.21 | 55.35 |
| grounded c=2 | 54.19 | 56.13 | 55.24 | 55.12 | 55.12 |
| grounded c=5 | 54.14 | 56.26 | 54.73 | 55.21 | 54.87 |

The linked row is identical to v1, which checks that the delta dump left the baseline postings unchanged.

R@1000 at c = 5:

| Document variant | REL | GPT-4 | Mixtral | E1-occ v2 | PRF score |
|---|---:|---:|---:|---:|---:|
| linked | 70.82 | 74.81 | 73.39 | 72.19 | **73.46** |
| alias c=5 | **70.99** | 75.35 | **73.96** | **72.43** | 73.29 |
| unfiltered c=5 | 70.83 | 75.01 | 73.50 | 72.19 | **73.46** |
| grounded c=5 | **70.99** | **75.40** | 73.90 | **72.43** | 73.21 |

Statistics, grounded v2 (c=5) vs controls (difference in nDCG@10 points; p_rand, p_Holm within each query side):

| Query side | vs linked | vs alias c=5 | vs unfiltered c=5 |
|---|---|---|---|
| REL | +0.48 (0.31, 0.62) | +0.08 (0.75, 0.75) | +1.07 (0.18, 0.53) |
| GPT-4 | −0.62 (0.32, 0.64) | +0.18 (0.13, 0.38) | −0.42 (0.56, 0.64) |
| Mixtral | −0.39 (0.59, 1.0) | +0.40 (0.31, 0.94) | −0.26 (0.75, 1.0) |
| E1-occ v2 | −0.25 (0.55, 1.0) | −0.06 (1.0, 1.0) | +0.16 (0.73, 1.0) |
| PRF score | −0.60 (0.08, 0.24) | −0.10 (0.50, 0.99) | +0.14 (0.83, 0.99) |

Mechanism diagnostics (positive entity-ID matches per judged document, relevant / non-relevant):

| | linked | alias c=5 | unfiltered c=5 | grounded c=5 |
|---|---|---|---|---|
| REL | 0.580 / 0.461 (×1.26) | 0.658 / 0.522 (×1.26) | 0.593 / 0.465 (×1.27) | 0.659 / 0.523 (×1.26) |
| GPT-4 | 1.957 / 1.142 (×1.71) | 2.181 / 1.266 (×1.72) | 1.993 / 1.152 (×1.73) | 2.190 / 1.273 (×1.72) |

Budget B = 1,940,031 added candidates (1,909,504 positive grounded postings). Unsupported-addition rate: alias 0%, grounded 0%, unfiltered 95.4%.

Blinded review of v2 (same protocol as §7.2: first 45 shuffled items of the blinded sample, not included, variant hidden, unblinded afterwards; `results_v2/review_judgements.json`). This review was done on 25 Nov in a new session after the 23 Nov session was cut off, and the judge saw only the stored first ~1,500 characters of each document:

| Variant | v1 precision | v2 judged correct | v2 precision |
|---|---:|---:|---:|
| alias | 38% | 6 / 16 | 38% |
| grounded | 38% | 6 / 16 | 38% |
| unfiltered | 15% | 3 / 13 | 23% |

The v2 rules removed the v1 error types they targeted (header words, sentence-initial capitals, weekday and "Last Year"-style hubs), but other error types took their place: given-name and surname pages for people (*Artyom* for "Artyom Shestakov", *McKnight* from a photo byline), and the wrong sense of a name inside a longer name (*Millennium* for "Millennium Tower", *Mini* for "MacBook Mini", *Republic* for Plato's *Republic*, *Armed Response (1986 film)* for "Armed Response Teams"). Several grounded additions (*Moon*, *Cambrian*, *Fathom*, *Dublin*) had no mention in the excerpt the judge could see; the mention may be later in the 512-token window.

**v2 reading.**
- Tighter evidence **removed the v1 harm**: grounded c=5 now leads linked by +0.48 for REL (v1: −1.63) and ties alias-only at every query side (largest gap 0.40, all p ≥ 0.13).
- It still **does not improve ranking**: no grounded-vs-control difference is significant. For the LLM and E1 query sides grounded stays 0.25–0.62 below linked-only. Gains appear only in R@1000 (+0.6 GPT-4, +0.5 Mixtral, +0.2 REL/E1).
- The relevant:non-relevant match ratio stays flat (×1.26 → ×1.26, ×1.71 → ×1.72). v2 adds matches to relevant and non-relevant documents in the same proportion, so the extra matches carry no ranking signal on average. This fits the unchanged 38% precision.
- v2 was designed after looking at v1 errors (not test metrics) and is labelled exploratory. Its null result therefore does not need a multiple-testing caveat, but its small REL gain should not be read as a confirmed effect.

## 8. Conclusions for Extension 2

1. **RQ2 (does evidence-grounded repair beat linked-only, alias-only and unfiltered expansion at a fixed budget?): not supported.** In the pre-registered v1, grounded repair is *worse* than linked-only and alias-only (REL −1.63, GPT-4 vs alias −1.41, p < 0.05 unadjusted, not significant after Holm). In the exploratory v2 it is statistically indistinguishable from every control on every query side.
2. **The mechanism works but the evidence is too noisy.** Repair adds 13–23% more entity matches to relevant documents, but adds them to non-relevant documents at the same rate. In two blinded reviews only about 38% of additions were correct. A string mention in the visible text is not enough evidence that the document is about that entity.
3. **Recall is the only consistent gain:** repaired documents raise R@1000 by about 0.5–0.6 points for the LLM query sides in both versions. Repair might therefore help a reranking pipeline more than it helps top-10 precision. We did not test this.
4. **Storage cost** is moderate. Grounded c=5 has 16.29M (v1, +28%) and 14.66M (v2, +15%) entity postings, against 12.75M linked. The entity part of the index grows in the same proportion (6 bytes/posting: 76.5 MB → 97.8 MB in v1 and 88.0 MB in v2; `results*/test.json`, `storage`). The word part is unchanged.
5. **What would be needed next:** a real disambiguation step, i.e. link probability p(e|m) from Wikipedia anchors (REL's tables, several GB), or retraining DyVo on repaired documents so the document head learns to down-weight noisy additions. The local-evidence-span scoring ablation (challenge 5) also remains open.
6. **Limitations:** frozen model, single seed, 42 test topics, a single non-independent judge for the reviews, and v2 designed after seeing v1 errors.

## 9. Follow-up (25 Nov): comparison with baseline DyVo
See `result.md`. With K and α chosen by topic-disjoint CV, both extensions together reach nDCG@10 55.03 (+1.36 over DyVo, not significant) and R@1000 73.32 (+2.99, p_Holm = 0.002). E2 contributes little on its own (+0.47 nDCG@10, +0.18 R@1000, not significant); interpolating repaired and linked scores gives significant R@1000 gains on top of GPT-4 (+0.23, p_Holm = 0.046) and, combined with E1, +1.33 (p_Holm = 0.009).
