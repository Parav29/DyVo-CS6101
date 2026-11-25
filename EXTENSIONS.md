# DyVo project extensions

**Status:** Proposed extensions for implementation after the DyVo reproduction baseline is frozen.  
**Last updated:** 23 November 2026.

This file records the extensions selected for the DyVo course project. They modify the entity-retrieval parts of the original system while keeping DyVo's learned sparse word representation, entity-scoring head, and sparse dot-product retrieval as the baseline.

The two primary extensions are:

1. **Utility-trained query entity selection** — improve which entity candidates are supplied for a query.
2. **Evidence-grounded document entity repair** — improve which entity IDs are assigned to documents and stored in the sparse index.

A third extension, **version-consistent entity refresh**, is conditional on obtaining suitable time-stamped data and relevance judgments.

> These are research proposals. They have not yet been shown to improve retrieval. Their value must be established through the controlled experiments described below.

## 1. Original DyVo pipeline

DyVo combines learned sparse word weights with sparse entity-ID weights.

```text
QUERY SIDE
Query
  → Transformer encoder
  → learned sparse word weights

Query
  → entity candidate generator
  → candidate entity IDs
  → frozen entity embeddings + DyVo query entity head
  → weighted entity-ID coordinates

DOCUMENT SIDE
Document
  → Transformer encoder
  → learned sparse word weights

Document
  → linked entity IDs
  → frozen entity embeddings + DyVo document entity head
  → weighted entity-ID coordinates

RETRIEVAL
Query word/entity vector
  × sparse document index
  → word overlap + shared canonical entity-ID overlap
  → ranked documents
```

The entity channel helps only when the query and a relevant document contain the **same canonical entity ID** with a useful positive weight. This creates two natural bottlenecks:

- The query candidate generator may return plausible entities that do not improve document ranking.
- A useful query entity cannot match a relevant document if that document lacks the corresponding entity posting in the index.

The two primary extensions address these bottlenecks on opposite sides of the same entity match.

---

## Extension 1 — utility-trained query entity selection

### Problem in the original system

DyVo accepts candidates from entity linkers, lexical or dense retrievers, and generated entity lists. These sources optimize linking, semantic similarity, or generation likelihood. A candidate can be semantically related to the query yet contribute nothing to retrieval because:

- the candidate does not occur in relevant document postings;
- it occurs in too many documents and has little discriminative value;
- it introduces a noisy entity match; or
- it consumes a limited candidate budget that could be used by a more useful entity.

### Core idea

Train a lightweight selector to predict a candidate entity's **document-retrieval utility**. The selector receives a cheap candidate pool and chooses the final entities sent to the existing DyVo query entity head.

### Pipeline placement

```text
Original:
Query → candidate generator → canonical entity IDs
      → DyVo query entity head → sparse retrieval

Modified:
Query → cheap candidate pool → utility-trained selector
      → selected canonical entity IDs → existing DyVo query entity head
      → existing sparse document index → sparse retrieval
```

### What it replaces

The selector replaces direct use of generic top-K candidates and, in the efficiency comparison, expensive online entity generation. It does **not** replace:

- the DyVo Transformer;
- the learned sparse word branch;
- the DyVo entity-weighting head;
- frozen entity embeddings; or
- the document index.

The original candidate sources remain baselines.

### Training signal

For each training query and candidate entity, estimate whether the entity improves the score margin between relevant and nonrelevant documents under a frozen DyVo model. A simple candidate-level signal is:

```text
query entity weight × (relevant-document entity weight − negative-document entity weight)
```

The project should compare retrieval-utility supervision against simpler controls:

- candidate-source score ranking;
- document-frequency ranking or pruning;
- teacher-list imitation;
- entity-occurrence labels; and
- random or fixed top-K selection.

All labels must be generated only from training data. Test-query relevance judgments must never be used to train the selector.

### Why it may help

The selector is trained for the final objective—ranking relevant documents—rather than generic entity similarity. It can also reduce inference cost by using a cheap candidate pool and a small local model instead of an LLM call at query time.

### Controlled experiment

Keep the following fixed across methods:

- candidate pool;
- final candidate budget K;
- DyVo checkpoint;
- document representations and index;
- training queries and negatives; and
- evaluation collection.

Compare:

1. original REL candidates;
2. original lexical/dense candidate ranking;
3. teacher-imitation selector;
4. utility-trained selector; and
5. utility-trained selector with adaptive K, if time permits.

### Measurements

**Effectiveness**

- nDCG@10 — primary metric
- nDCG@20
- Recall@1000
- candidate recall/coverage
- relevant versus nonrelevant entity overlap

**Efficiency**

- uncached query latency
- candidate-generation latency
- selector latency
- number of selected candidates
- peak GPU/CPU memory
- LLM/API calls and cost, when applicable

### Success criterion

At the same candidate pool and final K, utility supervision should improve document-retrieval metrics over generic ranking and teacher imitation. For an efficiency claim, it should retain most of the strongest candidate source's effectiveness while substantially reducing complete uncached query cost.

### Main risks

- A selector cannot recover an entity absent from the candidate pool.
- Utility labels may overfit a small set of training queries.
- Improvements may come from extra features or compute rather than the utility objective.
- A simple document-frequency rule may perform equally well.

### Fallback

If utility training does not improve effectiveness, retain the module as a cost-aware selector and test whether it reduces latency at a fixed acceptable effectiveness loss. If simple pruning matches it, report that negative result and use the simpler policy.

### Expected difficulty

**Medium.** The first experiment keeps all document vectors fixed, so it does not require rebuilding the index.

---

## Extension 2 — evidence-grounded document entity repair

### Problem in the original system

The paper compares several query-side candidate methods but uses linked entity candidates for documents. Better query candidates cannot produce an entity match when the relevant document lacks that entity in the sparse index.

Adding every remotely related entity to every document is not a valid solution: it increases index size, creates noisy matches, and can make any improvement a consequence of a larger posting budget.

### Core idea

Generate additional document entity proposals offline, retain only candidates supported by document evidence, and rebuild the document entity postings under a fixed index budget.

Acceptable evidence can include:

- an explicit mention;
- an alias or abbreviation;
- a supporting local text span; or
- a clearly recorded evidence relation for an implicit concept.

A generator's confidence alone is not treated as evidence.

### Pipeline placement

```text
Original:
Document text → linked entity IDs → DyVo document entity head
              → entity postings in the sparse index

Modified:
Document text → linked + retrieved entity proposals
              → evidence validation + posting-budget control
              → DyVo document entity head
              → rebuilt entity postings in a separate sparse index
```

An optional ablation restricts DyVo's entity scoring to the local evidence span. Candidate repair and local evidence scoring must be evaluated separately.

### What it replaces

The experimental index replaces linked-only document candidate lists with grounded, budgeted candidate lists. It does **not** replace:

- the word encoder;
- the query encoder;
- the DyVo scoring equation; or
- the sparse retrieval engine.

The original linked-only index remains the direct baseline.

### Why it may help

This extension repairs missing entity support on the document side. If a useful query entity is absent from a relevant document's representation because of linking failure, an evidence-supported posting can create the canonical-ID match DyVo needs.

### Core 2 × 2 experiment

| Query candidates | Linked document entities | Repaired document entities |
|---|---:|---:|
| Original REL candidates | Baseline | Document-side effect |
| Utility-selected candidates | Query-side effect | Combined effect |

This design separates the contribution of the query selector from the contribution of document repair.

Additional document-side controls:

1. linked entities only;
2. alias-only repair;
3. unfiltered entity expansion;
4. evidence-grounded repair; and
5. evidence-grounded repair with local-span scoring.

All variants must use the same visible document text and a matched total entity-posting or index-byte budget.

### Measurements

**Effectiveness**

- nDCG@10 — primary metric
- nDCG@20
- Recall@1000
- relevant-document entity overlap
- nonrelevant-document entity overlap
- unsupported entity-addition rate

**Efficiency and storage**

- total entity postings
- index size in bytes
- document encoding time
- offline proposal and validation time
- peak memory during index construction
- retrieval latency

### Success criterion

Evidence-grounded repair should improve nDCG or Recall over linked-only documents and unfiltered expansion at the same entity-posting/index budget. The added entities should also have a lower unsupported-addition rate than unfiltered expansion.

### Main risks

- There may be little missing-support headroom in CODEC.
- Alias-only repair may provide the same gain.
- Offline proposal generation may be expensive.
- Extra document text or extra postings may create an unfair comparison.
- Rebuilding several indexes may exceed available storage or compute.

### Fallback

If full repair is too costly, perform alias-based repair only and restrict the study to a diagnostic subset. If no measurable missing-support problem exists, reject the extension rather than claiming an unsupported benefit.

### Expected difficulty

**Hard.** Every document representation affected by the change must be regenerated and reindexed.

---

## Conditional extension 3 — version-consistent entity refresh

### Activation condition

Implement this extension only if the project obtains legitimate time-stamped entity catalogs, document snapshots, and document-relevance judgments. Hiding known entities at random can test unseen-vocabulary behavior, but it does not prove real temporal freshness.

### Problem in the original system

DyVo motivates a dynamic entity vocabulary, but adding an entity embedding row alone is insufficient. The new canonical ID must also be available to candidate retrieval and appear in relevant document postings. Changes to existing embeddings can make stored document weights stale.

### Core idea and placement

```text
Original:
Static entity catalog + embeddings + document links
  → one fixed DyVo sparse entity index

Modified:
Dated catalog change
  → update IDs and aliases
  → update candidate-search catalog
  → add or update entity embedding rows
  → identify affected documents
  → selectively rebuild their entity postings
  → publish a versioned index
```

### What it replaces

It replaces the assumption that a static catalog and entity index remain adequate as knowledge changes. It keeps the DyVo encoder and sparse scoring method.

### Controlled experiment

Compare:

1. stale index;
2. aliases updated only;
3. catalog and candidate retrieval updated;
4. selective document-posting refresh; and
5. full document-posting refresh.

Report results separately for new, changed, and stable entities/queries.

### Measurements

- nDCG@10 and nDCG@20
- Recall@1000
- regression on older queries
- number of documents and postings updated
- bytes written
- complete update time
- selective-refresh cost versus full-refresh cost

### Success criterion

Selective refresh should improve retrieval for genuinely new or changed entities while preserving performance on stable queries and costing materially less than a full index rebuild.

### Expected difficulty

**Hard**, with significant data risk. This is not one of the two required core extensions unless suitable temporal data becomes available.

---

## Implementation and evaluation order

1. Finish and verify the original DyVo and LSR-w reproduction.
2. Freeze checkpoints, preprocessing, query fields, candidate files, and evaluation settings.
3. Run candidate-coverage and document-support diagnostics.
4. Implement Extension 1 while keeping the document index fixed.
5. Evaluate Extension 1 against simple and teacher-imitation controls.
6. Implement Extension 2 and build separately named indexes for every document variant.
7. Run the 2 × 2 query/document experiment and required ablations.
8. Combine the two extensions only after recomputing any utility targets affected by the new index.
9. Attempt Extension 3 only if its temporal-data activation condition is satisfied.

## Minimum experiment matrix

| Run | Query candidate method | Document entity method | Purpose |
|---|---|---|---|
| B0 | None / word-only LSR-w | None | Word-only ancestor |
| B1 | REL | Linked-only | Original DyVo baseline |
| B2 | Best released query candidates | Linked-only | Strong released-candidate comparison |
| E1-C1 | Candidate-source ranking | Linked-only | Simple selector control |
| E1-C2 | Teacher imitation | Linked-only | Distillation control |
| E1 | Utility-trained selector | Linked-only | Extension 1 effect |
| E2-C1 | REL | Alias-only repair | Simple document control |
| E2-C2 | REL | Unfiltered expansion | More-candidates control |
| E2 | REL | Evidence-grounded repair | Extension 2 effect |
| E1+E2 | Utility-trained selector | Evidence-grounded repair | Combined effect |

Every run reports nDCG@10, nDCG@20, Recall@1000, end-to-end latency, relevant storage or offline cost, and the candidate/entity-support diagnostics needed to explain the result.

## Formal research questions

**RQ1.** Can a lightweight query entity selector trained on document-ranking utility outperform generic candidate ranking or teacher imitation when candidate pool, final K, and DyVo model are held constant?

**RQ2.** Can evidence-grounded document entity repair improve DyVo retrieval at a fixed entity-posting/index budget compared with linked-only documents, alias-only repair, and unfiltered expansion?

**RQ3.** Are Extensions 1 and 2 complementary, or do they solve the same subset of query-document entity mismatches?

**Conditional RQ4.** Can a version-consistent selective refresh improve retrieval for genuinely new entities at lower cost than a full document entity-index rebuild without harming stable queries?

## Final project narrative

DyVo adds dynamic entity coordinates to a learned sparse retriever, but a useful entity match requires the correct canonical entity ID on both the query and document sides. The project first reproduces the DyVo baseline, then tests two complementary hypotheses: select query entities according to their effect on document ranking, and repair missing document entity support using verifiable local evidence under a fixed index budget. Controlled query-side, document-side, and combined experiments determine whether either mechanism improves effectiveness or efficiency and identify when it fails.

## References and supporting project material

- Nguyen et al., **DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities**, EMNLP 2024: <https://aclanthology.org/2024.emnlp-main.45/>
- Official DyVo implementation: <https://github.com/thongnt99/DyVo>
- Detailed extension plan: [`research/DyVo-final-extension-plan.md`](research/DyVo-final-extension-plan.md)
- Full research-council report: [`research/DyVo-research-council-report.md`](research/DyVo-research-council-report.md)
- Professor-provided project paper: [`cs6101_DyVo.pdf`](cs6101_DyVo.pdf)

