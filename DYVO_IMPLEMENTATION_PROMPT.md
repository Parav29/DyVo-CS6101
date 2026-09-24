# DyVo Implementation Prompt: BUILD + SANITY-CHECK MODE (loop-driven, for Claude Code)

> **Paper:** *DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities*, Nguyen, Chatterjee, MacAvaney, Mackie, Dalton, Yates. EMNLP 2024. arXiv:2410.07722
> **Paper PDF:** `cs6101_DyVo.pdf` (in this project folder; read it first)
> **Official code:** https://github.com/thongnt99/DyVo (Apache-2.0)
> **HF paper page:** https://huggingface.co/papers/2410.07722
> **HF data repo:** `lsr42/dyvo_data` (a **dataset** repo)

You are building a DyVo codebase **on a low-resource machine**. The human will later move to a more powerful computer to download data, load real models, and train. Your job right now is:

1. Write all the code and files needed to implement and reproduce DyVo.
2. Prove the code works with **lightweight sanity checks** that use tiny, randomly initialized models and synthetic data.
3. Package every heavy step (downloads, real models, training, evaluation) as **scripts that are written and syntax-checked but NOT executed**, with a clear handoff guide.

You will be invoked **repeatedly in a loop**. Follow the Loop Protocol in Section 2 every iteration.

---

## 0. RESOURCE RULES (highest priority, never break these)

This machine is not for heavy work. Violating these rules is worse than leaving a task unfinished.

**Forbidden right now:**
- Downloading pretrained model weights. That includes `distilbert-base-uncased` weights, MonoT5, LaQue, BLINK, DPR, Mixtral, and any other checkpoint.
- Downloading anything from `lsr42/dyvo_data`. **Listing** the file names and sizes via the HF API is allowed; downloading files is not. (The entity embedding files alone are 6–22 GB each.)
- Downloading or loading datasets through `ir_datasets` (MS MARCO, Robust04, WaPo/Core 2018, CODEC).
- Calling any LLM API (OpenAI, Anthropic, etc.) or starting a local LLM server.
- Running any of the real experiment configs in `lsr/configs/experiment/` beyond **config composition** (`--cfg job`, see Level 0 below).
- Any single process running longer than ~5 minutes, using more than ~3 GB RAM, or requiring a GPU.

**Allowed:**
- Cloning the GitHub repo.
- Installing Python packages. Install **CPU-only PyTorch** to avoid multi-GB CUDA downloads: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. The big-PC setup script will install the CUDA build later.
- Creating tiny randomly initialized models locally from a config (no download), saving them under `smoke_assets/`, and running them on CPU.
- Generating synthetic data files, a synthetic tokenizer vocabulary, and a random entity embedding table.
- Short CPU runs of the official training entry point on the synthetic data (≤ 30 steps).
- **Exception to the download ban: small query-entity files.** You may download these files from `lsr42/dyvo_data`, because each is under 1 MB and lets tests use the authors' real formats: `{codec,robust04,wapo}/queries_entities_{gpt4,mistral,bm25,laque}.jsonl`, `{codec,robust04,wapo}/queries_test_train_entities_rel.jsonl`, and `codec/queries_test_train_entities_human.jsonl`. Use `hf download lsr42/dyvo_data --repo-type dataset --local-dir dyvo_data --include "<exact path>"`, one file at a time. Nothing else from that repo (in particular, not `entity_list.json` at 129 MB, not any `docs_entities.jsonl`, not anything in `knowledge_base/` or `dyvo_init/`).

**Enforce it in code:** `scripts/verify.sh` must export `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `WANDB_MODE=disabled`, and `CUDA_VISIBLE_DEVICES=""` before running anything. If any sanity check tries to reach the network, it should fail loudly. That's a bug to fix, not something to work around by going online.

---

## 1. Definition of done (for this build phase)

The build phase is **DONE** when all of the following are true:
- Every task in `PROGRESS.md` is `[x] DONE` or `[!] BLOCKED-EXTERNAL`.
- `bash scripts/verify.sh` exits 0 with all network access disabled.
- Every script under `scripts/big_pc/` passes `bash -n` / `python -m py_compile` and its `--dry-run` mode (prints what it would do, does nothing).
- `SANITY_REPORT.md` and `BIG_PC_HANDOFF.md` are written (Tasks S7.1, S7.2).

Only then print this exact line as the final line of your response, with nothing after it:

```
<promise>DYVO_BUILD_COMPLETE</promise>
```

Never print it otherwise. A loop runner watches for that string, and a false promise ends the loop with broken code.

---

## 2. Loop Protocol (every iteration)

**Step 1: Load state.** Read `PROGRESS.md`, `ERRORS.md`, `NOTES.md` from the project root. If they don't exist, this is iteration 1: start at Phase S0 (Section 6).

**Step 2: Pick one task.** Take the first `[ ]` or `[~]` task in `PROGRESS.md` whose dependencies are `[x]`. Mark it `[~]`. Do one task per iteration (two if both are small).

**Step 3: Do the work.** Before writing code for a component, read the matching official-repo file (Section 4).

**Step 4: Verify.** Run the task's own check, then `bash scripts/verify.sh` once it exists. A task is done only when its check passes.

**Step 5: Handle the result.**
- **Pass:** mark `[x]`; add one line to `NOTES.md`: what changed + the command + the key output line as evidence.
- **Fail:** follow the Error-Recovery Ladder.

**Step 6: Commit.** `git add -A && git commit -m "iter N: <task id> <result>"`.

**Step 7: Report.** End with a 3–6 line status: iteration number, task, result, next task. Add the promise line only if Section 1 is fully met.

### Error-Recovery Ladder

Log every failure in `ERRORS.md`:

```
## E<id> | task <task-id> | attempts: <n> | status: open|fixed|blocked
- symptom: <exact error line; first and last relevant traceback lines>
- hypothesis: <what you think is wrong>
- tried: <what you changed>
- result: <what happened>
```

Search `ERRORS.md` for the same symptom before trying a fix. Never repeat something already under "tried".

1. **Attempts 1–2:** read the full traceback, form a specific hypothesis, make the smallest fix, re-run.
2. **Attempt 3:** stop patching. Re-read the paper section and the official file. Write a 10–30 line reproduction script or pytest that isolates the failure; fix against it.
3. **Attempt 4:** take a different approach (different package version, different design, or `git revert` to the last green commit and redo the task differently).
4. **Attempt 5+:** if the only way forward breaks the Resource Rules (needs real weights, real data, a GPU, an API key), mark `[!] BLOCKED-EXTERNAL`, and write the exact next step for the human in `NOTES.md` under `## Deferred to big PC`. If it's an internal bug, mark `[!] BLOCKED-INTERNAL` with your diagnosis and come back to it later.

### Hard rules
- Never weaken, skip, or delete a test to make it pass. No `pytest.skip`/`xfail` to hide failures.
- Never write `try/except: pass` around a real error.
- Never present sanity-run metrics as real results. Numbers from random tiny models are meaningless for retrieval quality and must be labeled "sanity only" wherever they appear.
- Never change the method to make code simpler without logging it in `NOTES.md` under `## Deviations from paper`.
- If two iterations in a row make no progress on the same task, the next one must use ladder step 3 or 4.

---

## 3. What DyVo is (summary; read the PDF for detail)

LSR represents queries and documents as sparse weighted bags of vocabulary items, scored by dot product from an inverted index. BERT's ~30k word pieces split entities into fragments ("NFTs" → `n, ##ft, ##s`). DyVo adds millions of Wikipedia entities as extra vocabulary dimensions, scoring only a small **candidate** set per text.

- **Backbone:** `distilbert-base-uncased`, hidden size 768 (on the big PC). For sanity checks, a tiny random DistilBERT (Section 5).
- **Query encoder (MLP, Eq. 3):** per-token scalar weight `log1p(relu(W·h_j + b))`, masked for padding/special tokens, no expansion. `lsr/models/mlp_emlm.py::TransformerMLPeMLMSparseEncoder`.
- **Document encoder (MLM, Eq. 2):** SPLADE-style `max_j log1p(relu(logit_ij))`, with expansion. `lsr/models/mlm_emlm.py::TransformerMLMeMLMSparseEncoder`.
- **DyVo head (Eq. 4):** for each candidate entity, `λ_ent · max_j log1p(relu(e_i · h_j))`. The code applies `ent_weight` (λ_ent) to the logits *before* ReLU; match the code and note the difference.
- **λ_ent:** trainable scalar, initialized to **0.05**. Prevents entity collapse (Appendix A.2).
- **Entity embeddings:** precomputed and frozen. If entity dim ≠ backbone hidden size, a trainable linear projection maps it.
- **Entity IDs:** entity k has vocabulary ID `30522 + k`. `BertEntityEmbedding` **hardcodes** 30522, while `DynamicDataCollator` uses `tokenizer.vocab_size`. These must agree, so **every tiny test model and tokenizer must keep vocab size 30522**.
- **Joint score (Eq. 5):** word dot product + entity dot product, computed in `sparse_dot_product` (`lsr/losses/entity_distil_kl_loss.py`) by matching IDs; no vocab-size dense vector is built.
- **Candidates:** documents use REL-linked entities; queries use REL, BM25 top-20, LaQue top-20, LLM-generated (Prompt A.2, Mixtral/GPT-4), or human annotations (CODEC). `lsr/entity/bert_candidates.py::BERTCandidate` reads precomputed JSONL: `{"id": "<text id>", "entities": [<int>, ...]}`.
- **Training (big PC only):** start from `dyvo_data/dyvo_init`, 100k steps, batch 16, lr 5e-7, fp16, KL distillation from MonoT5-3B scores, BM25 negatives, query len 50, doc len 512, L1 on document word weights ∈ {1e-3, 1e-4, 1e-5}, no L1 on entities or queries.
- **Evaluation:** nDCG@10, nDCG@20, R@1000 via `ir_measures`. TREC datasets use the description field; CODEC uses the query field.

---

## 3b. LLMs in the paper, and what the authors already provide (verified)

These facts were checked against the paper, the official code, and the file listing of `lsr42/dyvo_data`. Treat them as the starting point, but confirm them yourself in S5.0.

**Which LLMs the paper uses, and whether they're free:**

| Model | Used for | Open-source? | How to run it yourself |
|---|---|---|---|
| Mixtral-8x7B-Instruct-v0.1 | (1) generating the 10k synthetic CODEC training queries (Prompt A.1); (2) generating query entity candidates (Prompt A.2) | Yes, open weights, Apache 2.0 | Hugging Face `mistralai/Mixtral-8x7B-Instruct-v0.1`, or Ollama `mixtral:8x7b` (≈26 GB, 4-bit; needs ≈32 GB RAM/VRAM). Ollama serves an OpenAI-compatible API at `http://localhost:11434/v1`. |
| GPT-4 | Generating query entity candidates (Prompt A.2) | No, proprietary | Only through OpenAI's paid API (needs an API key). |

Every other model in the paper is open (DistilBERT, MonoT5-3B teacher, LaQue, BLINK, DPR, Wikipedia2Vec, REL linker), and their expensive outputs are precomputed in the data repo.

**What's already in `lsr42/dyvo_data` (so no LLM or API key is needed to reproduce the paper):**

- **Mixtral synthetic training queries for CODEC:** `codec/codec_mistral_topics.tsv` + `codec/codec_mistral_qrels.json`. Training query IDs look like `mistral_16`.
- **GPT-4 query entities** for all three datasets: `{codec,robust04,wapo}/queries_entities_gpt4.jsonl`.
- **Mixtral query entities** for all three: `{codec,robust04,wapo}/queries_entities_mistral.jsonl`.
- **BM25 and LaQue query entities** for all three: `queries_entities_bm25.jsonl`, `queries_entities_laque.jsonl`.
- **Human query entities** (CODEC only): `codec/queries_test_train_entities_human.jsonl`.
- **REL query entities** (train + test queries): `queries_test_train_entities_rel.jsonl`.
- **Entity name ↔ ID mapping:** `knowledge_base/entity_list.json` (129 MB).
- **MonoT5-3B teacher scores:** `*_monot5_3b_scores.json`.

**Format of the LLM files** (one JSON object per line): `{"id": "<query id>", "text": "<query>", "entity_names": "<JSON-encoded list of names the LLM returned>", "entities": [<int entity ids>]}`. The `entities` list is often shorter than `entity_names`, because names not found in the knowledge base were dropped (out-of-vocabulary filtering, as the paper describes). `BERTCandidate` only reads `id` and `entities`, so the extra fields are harmless.

**How the authors used LLM entities (important for correct reproduction):**

1. **Test queries only, at inference time.** The GPT-4/Mixtral/BM25/LaQue files cover only the evaluation topics (e.g. 42 CODEC queries), not the training queries. Training always uses REL entities; the paper's Table 3 caption says GPT-4 candidates were used on queries for inference. So the LLM calls needed are ~340 in total (250 Robust04 + 50 Core 2018 + 42 CODEC), not thousands.
2. **The official configs are not wired to these files.** Every experiment config, including the Table 3 embedding-ablation configs that the paper says used GPT-4 candidates, points `query_ent_path` at `queries_test_train_entities_rel.jsonl`. There's also no LLM-calling code in the repo. So to reproduce Table 2/3 rows, you need a **merged candidate file**: REL entities for training query IDs + LLM entities for test query IDs (Task S5.3). Record this as a finding in `NOTES.md`.
3. **Entity ID 0 is a real entity** (e.g. "United States" maps to 0 in the GPT-4 files), and the collator also pads with ID 0 (masked). Padding is safe only because masked slots get weight 0; there's a test for this in S2.3.

**Known gap in the data repo:** the Robust04 configs reference `dyvo_data/robust04/docs.tsv`, which is **not** in `lsr42/dyvo_data`. It must be created on the big PC from the licensed corpus; `lsr/preprocess/export_robust04_docs.py` looks like the relevant script. Verify and write this into the handoff.

**Bottom line for the human:** you need **no paid API key and no local LLM** to reproduce every result in the paper, because the authors ship all LLM outputs. You only need an LLM if you want to regenerate candidates yourself, test a different LLM, or run DyVo on new queries. For that, Mixtral via Ollama/HF is free (needs a strong machine); GPT-4 needs a paid OpenAI key; a smaller open model (e.g. Llama 3.1 8B, Qwen2.5 7B/14B) runs on much less hardware but counts as a deviation and will likely score lower.

---

## 4. Official-repo map

| Concept | File(s) |
|---|---|
| Training entry (Hydra) | `lsr/train.py`, `lsr/configs/config.yaml` |
| Experiment configs (43) | `lsr/configs/experiment/*.yaml`. `qmlp_dmlm_*` = LSR-w baseline, `qmlp_dmlm_emlm_*` = DyVo; the embedding type is in the name (`laque`, `blink`, `wiki2vec`, `dpr`, `jds`, `bert` = token aggregation) |
| Dual encoder + checkpoint loading | `lsr/models/__init__.py` (`DualSparseEncoder`) |
| Query / doc encoders with entities | `lsr/models/mlp_emlm.py`, `lsr/models/mlm_emlm.py` |
| Entity embedding lookup | `lsr/models/entity_emb.py` |
| Candidate lookup | `lsr/entity/bert_candidates.py` |
| Batching + entity padding | `lsr/datasets/text_entity_collator.py` (`DynamicDataCollator`) |
| Data file readers (local TSV/JSON or `irds:`) | `lsr/utils/dataset_utils.py` |
| Loss + ID-matching dot product | `lsr/losses/entity_distil_kl_loss.py`, `lsr/losses/regularizer.py` |
| Trainer + eval loop | `lsr/trainer/__init__.py` |
| Tokenizer wrapper | `lsr/tokenizer/__init__.py` (`HFTokenizer`, calls `AutoTokenizer.from_pretrained(name)`, so a local directory path works) |

Data paths the real configs expect (for the big-PC scripts; do not download now): `dyvo_data/knowledge_base/entity_embs_{laque,blink,dpr,wiki2vec,jds,bert_agg}.pt`, `dyvo_data/dyvo_init/{query,doc}_encoder/model.safetensors`, and per dataset (`codec/`, `robust04/`, `wapo/`): topics `.tsv`, qrels `.json`, `*_monot5_3b_scores.json`, `docs_entities.jsonl`, `queries_test_train_entities_rel.jsonl`.

Approximate sizes in `lsr42/dyvo_data` (total ≈ 94.7 GB): `knowledge_base/` ≈ 92.9 GB (`entity_embs_blink.pt` 21.6 GB; `laque`, `dpr`, `jds`, `bert_agg` 16.2 GB each; `wiki2vec` 6.33 GB; `entity_list.json` 129 MB); `wapo/` ≈ 578 MB; `robust04/` ≈ 471 MB; `codec/` ≈ 177 MB. The per-dataset folders are mostly `docs_entities.jsonl` (123–516 MB); all query-entity files are under 1 MB.

---

## 5. Sanity-check design (how to test without real models or data)

### Tiny assets (built locally, no downloads) → `DyVo/smoke_assets/`
- **Tokenizer:** create a WordPiece `vocab.txt` with exactly **30522** entries: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, then the words used in the synthetic texts, then filler tokens (`[unused0]`...) to reach 30522. Save a `BertTokenizerFast`/`DistilBertTokenizerFast` built from it with `save_pretrained("smoke_assets/tiny-tokenizer")`. Assert `vocab_size == 30522`.
- **Model:** `DistilBertForMaskedLM(DistilBertConfig(vocab_size=30522, dim=32, hidden_dim=64, n_layers=1, n_heads=2, max_position_embeddings=128))`, randomly initialized, `save_pretrained("smoke_assets/tiny-distilbert")`. This is a few MB. Both encoders load it by passing this path as `tf_base_model_name_or_dir` (`AutoModel`/`AutoModelForMaskedLM.from_pretrained` accept local dirs).
- **Entity table:** random `torch.randn(200, 32)` saved as `smoke_assets/entity_embs_tiny32.pt`, plus `entity_embs_tiny16.pt` (200 × 16) to exercise the projection path. Also write `smoke_assets/entity_names.json` mapping IDs 0–199 to fake names (`"Entity_0"`, ...) plus a few realistic ones (`"Non-fungible token"`, `"Bitcoin"`, `"American Revolutionary War"`) for the LLM-parser tests.
- **Init checkpoints:** save the tiny dual encoder's state dict as `smoke_assets/dyvo_init/{query,doc}_encoder/model.safetensors` so the `query_encoder_checkpoint`/`doc_encoder_checkpoint` loading path is exercised.

### Synthetic data → `DyVo/smoke_data/`
Match the **exact formats** the real loaders read (inspect `lsr/utils/dataset_utils.py` and the dataset classes first; record the formats in `NOTES.md`): ~20 documents, ~10 training queries with MonoT5-style score JSON, 5 test queries, qrels, and `docs_entities.jsonl` / `queries_entities.jsonl` candidate files (some texts with 0 entities, some with several). Use local file paths, never `irds:`.

### Five levels of sanity checks
- **Level 0: static.** `python -m py_compile` on every `.py`; `bash -n` on every script; import every module under `lsr/models`, `lsr/losses`, `lsr/datasets`, `lsr/trainer`. **Config composition for all 43 real experiment configs** with `python -m lsr.train +experiment=<name> --cfg job` (prints the composed config without running `train()`; nothing is loaded). Every config must compose; record any that don't.
- **Level 1: unit tests** (pytest, tiny assets): the component tests in Phase S2.
- **Level 2: behavioral invariants** on the tiny model: all output weights ≥ 0; masked entities have weight exactly 0; word IDs < 30522 ≤ entity IDs; `sparse_dot_product` equals a dense reference; gradients reach both encoders, the projection, and `ent_weight`; **overfit test**: on one fixed batch, 50–100 optimizer steps (lr ~1e-3) drive the KL loss down substantially (e.g. ≥ 50% from the start). A model that can't overfit one batch has a wiring bug.
- **Level 3: end-to-end smoke.** Run the official `python -m lsr.train` with smoke configs for ≤ 30 steps on CPU: training runs, a checkpoint saves, prediction writes a run file, and `ir_measures` prints nDCG@10 / nDCG@20 / R@1000. The values are meaningless; only the pipeline completing matters.
- **Level 4: big-PC scripts dry-run.** Every `scripts/big_pc/*` script supports `--dry-run` and passes it.

---

## 6. Task plan (seed `PROGRESS.md` with this in iteration 1)

### Phase S0: Scaffolding
- **S0.1** Create `PROGRESS.md`, `ERRORS.md`, `NOTES.md` (sections: `## Environment`, `## Code ↔ paper map`, `## Data formats`, `## LLMs and API keys`, `## Deviations from paper`, `## Deferred to big PC`, `## Sanity log`). Fill `## LLMs and API keys` from Section 3b right away, then update it with what you confirm in S5.0. `git init`. *Check:* files exist.
- **S0.2** Create the slash command files from Appendix A. *Check:* files exist.
- **S0.3** Record environment in `NOTES.md`: OS, Python, `free -g`, `df -h`, GPU present or not. *Check:* filled.

### Phase S1: Code + environment
- **S1.1** Clone `https://github.com/thongnt99/DyVo` into `./DyVo`. *Check:* `DyVo/lsr/models/mlm_emlm.py` exists.
- **S1.2** Create a Python 3.10 venv (or 3.9, as the README uses 3.9.12). Install CPU torch, then `requirements.txt` plus deps the code actually imports but doesn't list (grep all imports; expect at least `wandb`, `pandas`, `safetensors`, `omegaconf`, `faiss-cpu`, `accelerate`, `pytest`). Pin `transformers` to a version that still accepts the configs' `evaluation_strategy` key and the `HFTrainer` overrides (try `4.40.2` first). Write the exact working versions to `requirements-lock.txt`. *Check:* imports succeed.
- **S1.3** Read the paper and every file in Section 4. Write the code ↔ paper map in `NOTES.md`, including discrepancies. Also check and note these known issues: the `entity_cls=True` branch in `mlp_emlm.py` uses `.unsqueeze(0)` vs `.unsqueeze(1)` in `mlm_emlm.py` (unused by default); some configs contain `sparse_aspect: Flase` (a truthy string; check whether it's read); `loss.normalize: "std"` falls into the L2-normalize branch rather than softmax (keep it for reproduction, but note it). *Check:* section present.

### Phase S2: Tiny assets + unit tests (Levels 0–1)
- **S2.1** `DyVo/smoke_assets/build_assets.py` builds everything in Section 5's "Tiny assets", offline, in under 30 s. *Check:* runs with network env vars set to offline; `vocab_size == 30522` asserted.
- **S2.2** `scripts/check_configs.sh`: composes all 43 real experiment configs with `--cfg job`. *Check:* all compose, or the failures are logged with a reason.
- **S2.3** Tests in `DyVo/tests/` (tiny assets only):
  - `test_entity_emb.py`: correct rows for offset IDs, shape `(B, n_ent, dim)`, no grad into the table.
  - `test_dyvo_head.py`: both encoders' entity weights equal a hand computation within 1e-5; masking; non-negativity; output IDs = word IDs ++ entity IDs.
  - `test_projection.py`: entity dim 16 → projection exists and is trainable; entity dim 32 (= hidden) → no projection.
  - `test_sparse_dot.py`: `sparse_dot_product` equals the scatter-to-dense reference; no word/entity ID collisions.
  - `test_lambda_ent.py`: `ent_weight` is an `nn.Parameter` initialized from config (0.05), gets a gradient, and some entity weights are nonzero at init.
  - `test_collator.py`: `DynamicDataCollator` padding, masks, offset by vocab size, zero-entity items.
  - `test_candidates.py`: `BERTCandidate` reads the smoke JSONL, deduplicates, returns ints and masks, and returns empty lists for unknown IDs. Also reads a line in the authors' LLM-file format (with extra `text` and `entity_names` fields) correctly.
  - `test_entity_zero.py`: a real entity with ID 0 in one text and a padding slot (also ID 0, masked) in another must not create a spurious match: the padded slot's weight is 0, so the score contribution is 0.
  - `test_loss.py`: `EntityDistilKLLoss` gives a finite scalar; gradients reach both encoders and `ent_weight`.
  - `test_checkpoint_load.py`: `DualSparseEncoder` loads the tiny safetensors init checkpoints; report missing/unexpected keys (loading uses `strict=False`, so assert the important keys actually loaded).
  *Check:* `pytest DyVo/tests -q` passes offline.

### Phase S3: Behavioral checks (Level 2)
- **S3.1** `DyVo/sanity/overfit_one_batch.py`: builds the tiny DyVo dual encoder, one fixed batch of 2 queries × (pos, neg), trains 50–100 steps with Adam, prints the loss curve and the count of nonzero entity weights per step. *Check:* the final loss is well below the initial loss (state the threshold you used), and entity weights are not all zero at the end.
- **S3.2** Collapse probe: the same script with `--lambda-init 1.0` and `--no-train-lambda`. Log whether entities collapse (all zero) at toy scale. It's fine if it doesn't reproduce at this scale; say so honestly in `NOTES.md`. *Check:* both runs logged.

### Phase S4: End-to-end smoke (Level 3)
- **S4.1** Build the synthetic data in `DyVo/smoke_data/` (Section 5). *Check:* the official readers load it (small script that calls them and prints counts).
- **S4.2** Add `lsr/configs/experiment/smoke_dyvo.yaml` and `smoke_lsrw.yaml`: tiny model/tokenizer paths, tiny entity table with `entity_dim: 32`, smoke data paths, `max_steps: 20`, batch 2, `fp16: False`, `dataloader_num_workers: 0`, `report_to: none`, `per_device_eval_batch_size: 2`, short max lengths. Base them on the real CODEC DyVo config so the same code paths run. *Check:* `python -m lsr.train +experiment=smoke_dyvo` completes offline in < 3 min, writes a run file, and prints the three metrics. Same for `smoke_lsrw`.
- **S4.3** `scripts/verify.sh` (project root): `set -euo pipefail`; offline env vars from Section 0; Level 0 static checks; config composition; `pytest`; the overfit check; both smoke runs; asserts that the run files exist and are non-empty. Total runtime under ~5 min. *Check:* `bash scripts/verify.sh; echo $?` → 0.

### Phase S5: LLM entity candidates (use the authors' files first; generator is optional)
Read Section 3b first. The primary path uses the authors' precomputed LLM outputs; no LLM or API key is needed.
- **S5.0** Confirm Section 3b. List `lsr42/dyvo_data` file names and sizes via the HF API (`HfApi().list_repo_tree(..., repo_type="dataset", recursive=True)`, no downloads) and save the listing to `NOTES.md`. Download only the small query-entity files allowed in Section 0 into `dyvo_data/`. For each dataset and candidate type, record in `NOTES.md` under `## LLMs and API keys`: number of queries covered; whether they're test-only or train+test (compare IDs with the REL file); average `len(entity_names)` vs `len(entities)` (the OOV drop rate); and whether any test query is missing. *Check:* table present in `NOTES.md`.
- **S5.1** `DyVo/lsr/entity/merge_candidates.py`: builds a query candidate file with REL entities for training query IDs and the chosen source (`gpt4`, `mistral`, `bm25`, `laque`, `human`) for test query IDs. Output: `dyvo_data/<dataset>/queries_test_train_entities_<source>.jsonl` in the `{"id", "entities"}` format. Report any test query with no LLM entry (fall back to REL for it and log how many). *Check:* runs on the downloaded small files; output line counts match the REL file; every test ID's entities come from the chosen source.
- **S5.2** Add experiment configs for the Table 2 rows (`qmlp_dmlm_emlm_laque_<dataset>_..._<source>.yaml`, copied from the REL configs with only `query_ent_path` changed) and for the Table 3 rows (each embedding type with the GPT-4 merged file). Note in each file's header that the official configs used REL here, and that this change follows the paper's text. *Check:* all new configs compose with `--cfg job`.
- **S5.3** `tests/test_merge_candidates.py`: tiny synthetic REL + LLM files; checks train IDs keep REL, test IDs take the LLM source, missing test IDs fall back, and the output reads through `BERTCandidate`. Also run once against the real small files if downloaded. *Check:* passes offline.
- **S5.4 (optional, for regenerating or new queries)** `DyVo/lsr/entity/generate_entities_llm.py`: reads a queries file; builds the prompt from the **exact** Prompt A.2 template in the paper appendix (both few-shot examples verbatim); calls a pluggable backend selected by env vars (`openai`-compatible endpoint, which covers OpenAI GPT-4, Ollama at `http://localhost:11434/v1`, and vLLM; or `anthropic`), with keys only from env vars; parses the returned list robustly; maps names → IDs using `knowledge_base/entity_list.json` (inspect its structure on the big PC; for now, code against a small synthetic file with the same shape as your best guess and make the loader fail with a clear message if the real shape differs); drops OOV names and logs the OOV rate; writes output in the **same format as the authors' files** (`id`, `text`, `entity_names`, `entities`). Include `--backend mock`. **No real API calls now.**
- **S5.5 (optional)** `tests/test_llm_entities.py`: with the mock backend, test valid JSON lists, lists wrapped in prose or code fences, malformed output, duplicates, case differences, and OOV names; check the output reads through `BERTCandidate`. When the real small GPT-4 files are available, also check that your parser recovers the same name list from their `entity_names` field. *Check:* passes offline.

### Phase S6: Big-PC scripts (write + dry-run only; do NOT execute for real)
Every script supports `--dry-run` (prints each command/action, performs none), uses `set -euo pipefail` (bash) and clear logging, and is idempotent (skips work already done).
- **S6.1** `scripts/big_pc/00_setup_env.sh`: create the env, install **CUDA** torch, install the locked requirements, verify `torch.cuda.is_available()`.
- **S6.2** `scripts/big_pc/01_list_and_download_data.py`: lists `lsr42/dyvo_data` files with sizes (HF API, `repo_type="dataset"`), checks free disk against what's selected, then downloads selectively: `hf download lsr42/dyvo_data --repo-type dataset --local-dir dyvo_data --include <patterns>`. Flags: `--datasets codec,robust04,wapo`, `--embeddings laque,wiki2vec,...`. The README's plain `huggingface-cli download lsr42/dyvo_data` is wrong (no `--repo-type dataset`, lands in the cache); don't copy it.
- **S6.3** `scripts/big_pc/02_check_data.py`: verifies every path the chosen experiment configs need exists; checks RAM vs embedding file size (LaQue ≈ 16.2 GB, BLINK ≈ 21.6 GB, DPR ≈ 16.2 GB, Wiki2Vec ≈ 6.3 GB are loaded fully into CPU RAM by `BertEntityEmbedding`; warn if RAM is too small and point to an optional `mmap=True` load); checks that `ir_datasets` can see each corpus (Robust04 and WaPo need licensed data placed manually; print the instructions); and runs an **ID alignment check** (max entity ID in candidate files < number of rows in the embedding table; spot-check known names if a name↔ID mapping exists).
- **S6.4** `scripts/big_pc/03_train.sh <experiment-name>`: runs `python -m lsr.train +experiment=<name> training_arguments.fp16=True` under `nohup`, logs to `logs/<name>.log`, and refuses to start if the same run is already going.
- **S6.5** `scripts/big_pc/04_collect_results.py`: parses logs/run files, recomputes nDCG@10/@20/R@1000 with `ir_measures`, and writes `results/results.csv` next to the paper's numbers (Section 7) with the difference.
- **S6.6** `scripts/big_pc/05_generate_entities.sh` (**optional**; not needed to reproduce the paper): runs the S5.4 generator with a real backend. Flags: `--backend ollama|openai|anthropic|vllm`, `--model <name>`, `--dataset`. Before running, it prints a warning that the authors' GPT-4/Mixtral files already exist, and that using a different model is a deviation. With `--backend openai`, it refuses to run unless `OPENAI_API_KEY` is set. With `--backend ollama`, it checks that the server responds and that the model is pulled.
- **S6.6b** In `02_check_data.py`, also check for `dyvo_data/robust04/docs.tsv` (not shipped in the data repo), and print how to build it from the licensed corpus.
- **S6.7** `scripts/big_pc/run_priority.sh`: the recommended order. First, CODEC LSR-w vs DyVo(REL, LaQue) at L1 = 1e-5. Then Robust04 and Core 2018. Then the L1 sweep (1e-3, 1e-4). Then LLM candidates using the authors' provided files and the S5.2 configs (Table 2). Then the embedding ablation (Table 3). Running `merge_candidates.py` is part of this script; calling an LLM is not.
*Check (all):* `bash -n` / `py_compile` pass, and every script's `--dry-run` exits 0 on this machine; add these dry-runs to `verify.sh`.

### Phase S7: Reports
- **S7.1** `SANITY_REPORT.md`: for each level (0–4), what was checked, the command, pass/fail, and key evidence (e.g. the overfit loss start → end, number of configs that composed). A clear statement that no real models or data were used and that no metric in this report says anything about retrieval quality.
- **S7.2** `BIG_PC_HANDOFF.md`: hardware needed (GPU with ≥ 24 GB recommended; an A100 was used in the paper), disk (roughly 25–80 GB depending on which embeddings; the full data repo is ≈ 95 GB), RAM (≥ 32 GB for LaQue, more for BLINK), licensed data to obtain (including building `robust04/docs.tsv`), the exact command sequence, the expected runtime per run, and what to check after the first run. It **must** include a section titled **"Do I need LLMs or API keys?"** written for the human in plain language, containing:
  1. The table from Section 3b of which LLMs the paper used and which are open (Mixtral: free, open weights on Hugging Face and Ollama; GPT-4: paid OpenAI API only).
  2. The answer: **no paid API key or local LLM is needed to reproduce the paper**, because the authors' data repo already contains the Mixtral training queries and the GPT-4, Mixtral, BM25, LaQue, and human entity candidates. List the exact files, with the S5.0 findings (queries covered, OOV drop rate).
  3. When you *would* need an LLM (regenerating candidates, testing another model, new queries), and the options: Mixtral via Ollama (`ollama pull mixtral:8x7b`, ≈26 GB, ≈32 GB RAM/VRAM), GPT-4 via OpenAI (set `OPENAI_API_KEY`; about 340 short calls for all test sets), or a smaller open model as a documented deviation. The exact `05_generate_entities.sh` command for each.
  4. The finding that the official configs are wired only to REL entities, and that `merge_candidates.py` + the S5.2 configs are what make the LLM rows reproducible.
  Also copy a short version of this section into `NOTES.md` under `## Deferred to big PC`.
- **S7.3** Final `bash scripts/verify.sh` green. Then check Section 1.

---

## 7. Reference numbers (for `04_collect_results.py` only; never written as results)

L1 = 1e-5, DistilBERT, LaQue embeddings:

| Method | Robust04 nDCG@10 | R@1k | Core18 nDCG@10 | R@1k | CODEC nDCG@10 | R@1k |
|---|---|---|---|---|---|---|
| LSR-w | 49.13 | 66.86 | 40.99 | 63.22 | 52.61 | 69.07 |
| DyVo (REL) | 51.19 | 68.56 | 43.72 | 63.56 | 53.40 | 70.60 |
| DyVo (Mixtral) | 52.97 | 69.28 | 43.80 | 68.27 | 54.90 | 73.20 |
| DyVo (GPT-4) | 54.39 | 70.86 | 43.06 | 68.57 | 56.46 | 74.47 |
| DyVo (Human) | – | – | – | – | 56.42 | 75.13 |

Other L1 levels, nDCG@10 (Robust04 / Core18 / CODEC): 1e-3: LSR-w 40.37 / 34.50 / 39.10, DyVo (REL) 41.52 / 37.50 / 42.67. 1e-4: LSR-w 47.69 / 38.94 / 50.54, DyVo (REL) 48.15 / 43.10 / 51.66.

Embedding ablation with GPT-4 candidates, nDCG@10 (Robust04 / Core18 / CODEC): Token Aggr. 51.35 / 41.63 / 53.44; DPR 48.68 / 40.26 / 53.04; JDS 51.21 / 44.29 / 55.08; Wiki2Vec 54.04 / 44.15 / 56.30; LaQue 54.39 / 43.06 / 56.46; BLINK 55.56 / 44.63 / 58.15.

---

## 8. Engineering standards

- Extend the official repo; don't rewrite it. New code goes in `DyVo/tests/`, `DyVo/sanity/`, `DyVo/smoke_assets/`, `DyVo/smoke_data/`, `DyVo/lsr/entity/generate_entities_llm.py`, `DyVo/lsr/configs/experiment/smoke_*.yaml`, `scripts/`.
- Every edit to an official file gets a `# DYVO-REPRO: <reason>` comment and a `NOTES.md` entry. Edits must not change behavior of the real configs (verify via Level 0 composition).
- Seed everything (`transformers.set_seed`, torch, numpy, random) and record seeds.
- Keep `verify.sh` fast and fully offline.

---

## Appendix A: Slash command files (create in S0.2)

### `.claude/commands/dyvo-iter.md`
```markdown
---
description: Run one DyVo build-phase loop iteration
---
Read DYVO_IMPLEMENTATION_PROMPT.md in full, especially Section 0 (Resource Rules). Execute exactly one iteration of the Loop Protocol (Section 2).
Extra focus for this iteration, if any: $ARGUMENTS
End with the 3–6 line status. Print the completion promise only if Section 1 is fully met.
```

### `.claude/commands/dyvo-fix.md`
```markdown
---
description: Focused error-fixing pass
---
Read DYVO_IMPLEMENTATION_PROMPT.md Sections 0 and 2, then ERRORS.md.
Pick the open error with the most attempts (or the one named here: $ARGUMENTS).
Apply the Error-Recovery Ladder at the step its attempt count requires. Do not repeat anything under "tried". Never break the Resource Rules to fix an error.
Re-run the failing check and bash scripts/verify.sh. Update ERRORS.md and commit.
```

### `.claude/commands/dyvo-verify.md`
```markdown
---
description: Run the offline verification gate and report
---
Run `bash scripts/verify.sh`. If it fails, add or update an ERRORS.md entry with the exact failing line and mark the related PROGRESS.md task [~]. Report only; change no code.
```

### `.claude/commands/dyvo-status.md`
```markdown
---
description: Summarize progress without changing anything
---
Read PROGRESS.md, ERRORS.md, NOTES.md. Report tasks done / in progress / blocked, open errors with attempt counts, and the "Deferred to big PC" list. Make no changes.
```

---

## Appendix B: Running the loop (for the human)

**Ralph-style loop plugin (autonomous):**
```
/ralph-loop "Read DYVO_IMPLEMENTATION_PROMPT.md and execute one iteration of its Loop Protocol." --completion-promise "DYVO_BUILD_COMPLETE" --max-iterations 40
```
Always keep a `--max-iterations` cap. Check the plugin's help for exact flag names on your version.

**Manual:** run `/dyvo-iter` repeatedly; `/dyvo-fix` when something breaks; `/dyvo-status` any time.

**Headless shell loop:**
```bash
mkdir -p logs
for i in $(seq 1 40); do
  out=$(claude -p "/dyvo-iter" --permission-mode acceptEdits 2>&1 | tee -a logs/loop.log)
  echo "$out" | grep -q "<promise>DYVO_BUILD_COMPLETE</promise>" && { echo "Build done at iteration $i"; break; }
done
```

**When you move to the big PC:** copy the whole project folder, then follow `BIG_PC_HANDOFF.md`. Start with `bash scripts/big_pc/00_setup_env.sh`, run each script with `--dry-run` first, then for real.
