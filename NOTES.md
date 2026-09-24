# NOTES

## Environment
- Build machine: Apple M4 (arm64), macOS 26.6.2, 16 GB RAM, ~14 GB free disk, **no CUDA GPU** (Apple Metal only; not used).
- Python: 3.10.20 in `.venv/` (activate with `source .venv/bin/activate`). CPU-only torch 2.2.2.
- Key pins (full list in `requirements-lock.txt`; wandb pinned to 0.16.6 because 0.30 rejects `Settings(start_method=...)`, E2): transformers 4.40.2, tokenizers 0.19.1, accelerate 0.30.1, hydra-core 1.3.2, omegaconf 2.3.1, datasets 2.19.1, ir-datasets 0.6.3, ir-measures 0.4.3, numpy 1.26.4, faiss-cpu 1.15.1, safetensors 0.8.0, wandb 0.16.6, pytest 9.1.1.
- Upstream code: `thongnt99/DyVo` @ `1b30d6c` (cloned into `DyVo/`, its `.git` removed so the outer repo tracks all edits; iter-0 commit is the pristine baseline).
- Seeds: `training_arguments.seed=42` (HF `set_seed`), sanity scripts/tests seed `torch`, `numpy`, `random` with 0 unless stated.

## Code ↔ paper map
| Paper | Code | Notes |
|---|---|---|
| Eq. 1 sparse dot product | `lsr/losses/entity_distil_kl_loss.py::sparse_dot_product` (training), `lsr/trainer/__init__.py::evaluation_loop_iterative_vectorized` (eval) | ID-matching over `(ids, weights)` pairs; no |V|-sized dense vector is built. |
| Eq. 2 MLM doc encoder (SPLADE) | `lsr/models/mlm_emlm.py::TransformerMLMeMLMSparseEncoder.forward` (word part) | `max_j log1p(relu(logits))` with pad/special tokens masked **before** relu; then top-k over non-zeros. |
| Eq. 3 MLP query encoder | `lsr/models/mlp_emlm.py::TransformerMLPeMLMSparseEncoder.forward` (word part) | `log1p(relu(W h_j + b))` masked by attention & special tokens; ids = input ids (no expansion). Duplicate tokens are summed at scoring (Eq. 3's indicator sum). Paper Eq. 3 omits log1p(relu); code has it. |
| Eq. 4 DyVo head | both `*_emlm.py` files, "ENTITY SCORES" block | code: `log1p(relu(λ·(e_i·h_j)))`, max over j; paper: `λ·max_j log1p(relu(e_i·h_j))`. See Deviations. Entity logits are computed against **all** positions incl. [CLS]/[SEP]/padding (no attention mask on h_j). |
| λ_ent trainable, init 0.05 | `self.ent_weight = nn.Parameter(config.entity_weight)`; experiment configs set `entity_weight: 0.05` | Each encoder has its **own** λ (query and doc). Class default is 0.01; configs override to 0.05. |
| Eq. 5 joint score | concatenation `ids = [word ids ++ entity ids]`, `weights = [word w ++ entity w]` then `sparse_dot_product` | Entity k ↔ id `30522 + k`. |
| Entity embeddings (frozen) | `lsr/models/entity_emb.py::BertEntityEmbedding` | Full table `torch.load`ed to CPU; lookup under `no_grad`; **hardcodes offset 30522**. |
| Projection when dims differ | `self.entity_proj = nn.Linear(entity_dim, hidden)` in both encoders | Created only if `entity_dim != hidden_size`. |
| Candidate retrieval | `lsr/entity/bert_candidates.py::BERTCandidate` | Reads `{"id","entities"}` JSONL; dedups with `set` (order not preserved); unknown id → `[]`. |
| Batching & padding | `lsr/datasets/text_entity_collator.py::DynamicDataCollator` | Pads entity lists with id 0 + mask 0, then adds `tokenizer.vocab_size` (30522). At least 1 slot per text (masked if empty). Docs use `d_max_length`; **queries also use `d_max_length`** in `pad_one_group` (see Discrepancies). |
| KL distillation from MonoT5-3B | `EntityDistilKLLoss` + `TripletIDDistilDataset` (`train_group_size=2`: 1 pos + 1 neg) | Student: log_softmax over [pos, neg]; teacher: `normalize="std"` in configs → L2-normalize of raw teacher scores (see Discrepancies). |
| L1 on doc words only, none on entities/queries | `L1(tok_ids=...)` masks ids ≥ 30522 when `reg_ent: False`; `q_regularizer.weight: 0.0` | Weight ramps quadratically to target over `T=50000` steps. |
| Training hparams | experiment YAMLs | lr 5e-7, 100k steps, batch 16 (HF default `per_device_train_batch_size` overridden? → see Discrepancies), warmup_ratio 0.1, q len 50, d len 512, fp16 via CLI. |
| Init from MSMARCO LSR | `DualSparseEncoder(query_encoder_checkpoint=..., doc_encoder_checkpoint=...)` | `load_state_dict(strict=False)` from `dyvo_data/dyvo_init/*/model.safetensors`. |
| Evaluation nDCG@10/20, R@1000 | `HFTrainer.predict` → `evaluation_loop_iterative_vectorized` → `ir_measures.calc_aggregate` | Brute-force scoring of all test docs, keeps top-10000 per query, writes `test_run.trec` + `test_result.json`. |
| Datasets | `configs/dataset/{codec,trec_robust04,trec_core18}.yaml` | CODEC uses `query` field; TREC uses `description` (matches paper). |

### Discrepancies / known issues found while reading
1. **`lsr/losses/__init__.py` imports 3 deleted modules** (`aspect_distil_kl_loss`, `sparse_aspect_kl_loss`, `tokent_aspect_kl_loss`) → `import lsr.losses` fails, so *no* training config can run from a fresh clone. Fixed (commented out, `# DYVO-REPRO`). Not referenced by any config.
2. **`TransformerMLPeMLMConfig` never stores `entity_cls`** → `self.config.entity_cls` raises `AttributeError` in the DyVo query encoder's forward. Fixed by storing it (default False = paper behavior; `# DYVO-REPRO`).
3. `entity_cls=True` branch: `mlp_emlm.py` uses `.unsqueeze(0)` (wrong: yields shape `1×B×H`), `mlm_emlm.py` uses `.unsqueeze(1)` (correct). Unused by default (False); left as is.
4. 24 configs contain `sparse_aspect: Flase` (truthy string). It is only stored as an extra config attribute; the only reader is `mlp_ebin.py`, which no DyVo config uses → harmless.
5. `loss.normalize: "std"` (39 of 42 configs using it) is not a recognized name: anything other than `"softmax"` falls into `F.normalize(labels, dim=1)` (L2 normalization of teacher scores, not a probability distribution). `KLDivLoss` then compares a log-prob with a non-normalized target. Kept for reproduction. **Consequence (found in S2.3, E1):** any negative teacher score makes the KL target negative → NaN loss. The real `*_monot5_3b_scores.json` must therefore hold non-negative scores (e.g. probabilities); `02_check_data.py` verifies this on the big PC.
6. `DynamicDataCollator.pad_one_group` truncates **queries with `d_max_length`** (512), not `q_max_length` (50). `q_max_length` is effectively unused by the DyVo collator. Kept (matches authors' runs); the LSR-w `DataCollator` does use `q_max_length` → a small asymmetry between LSR-w and DyVo.
7. Eval writes representations to `/scratch-shared/$USER/lsr-entities/...` (authors' cluster path) — cannot be created on most machines. Patched to honor `LSR_SCRATCH_DIR` env var, default unchanged (see Deviations/edits).
8. Eval uses `torch.autocast(device_type='cuda', dtype=float16)` — on CPU this only warns and is disabled.
9. `BertEntityEmbedding` hardcodes 30522 while the collator uses `tokenizer.vocab_size` → tiny tokenizer must have exactly 30522 entries.
10. `per_device_train_batch_size` is 128 in base `config.yaml`; the paper says batch 16 for fine-tuning. The experiment YAMLs **do not override it** → verify on big PC (could OOM at 512 tokens × 128 × 2 docs). Recorded as a big-PC check; `03_train.sh` passes `training_arguments.per_device_train_batch_size=16` to match the paper (see Deviations).
11. Base `training_arguments.dataloader_num_workers: 16`, `report_to: wandb`: smoke configs override to 0 / none.
12. **Robust04 `docs.tsv` (correction to prompt §3b):** the Robust04 training dataset config actually used by the experiment configs (`dataset/robust04_inparsv2_monot53b.yaml`) reads docs from `irds:disks45/nocr/trec-robust-2004`; its `dyvo_data/robust04/docs.tsv` line is commented out. Only the unused `robust04_entity_inparsv2_monot53b.yaml` references `data/robust04/docs.tsv`. So `docs.tsv` is **not required**; what is required is the licensed TREC disks 4&5 placed for `ir_datasets`. `lsr/preprocess/export_robust04_docs.py` can still build a `docs.tsv` if you prefer local TSV (switch the commented line).
13. **11 configs referenced a missing `dataset/trec_robust04_fold1`** (as eval_dataset): LSR-w Robust04 1e-5 and 1e-3, Table-3 Robust04 bert/blink/dpr/jds, and 5 legacy `qmlp_dmlm_emlm_robust04_*`/`qmlp_dmlm_emlm_codec_*_0.00001` configs. Added `lsr/configs/dataset/trec_robust04_fold1.yaml` → `irds:disks45/nocr/trec-robust-2004/fold1` (new file, `# DYVO-REPRO`). Used only for in-training monitoring; test = all 250 topics.
15. **Wiki2Vec configs had the wrong `entity_dim`** (would crash on the first forward pass): Wiki2Vec tables are 300-d (6.33 GB = 5.28M × 300 × 4 B, same row count as the 768-d LaQue table), but `qmlp_dmlm_emlm_wiki2vec_{codec,robust04}_*` set the query encoder to 768 and `..._wapo_*` set both encoders to 768, so no `entity_proj` is created and `hidden(768) @ entity(300)` fails. Fixed to 300 in the 3 official configs + their 3 GPT-4 copies (`# DYVO-REPRO`); `scripts/check_entity_dims.py` (in verify.sh) now checks every DyVo config: 53 checked, 0 mismatches. (BLINK 21.6 GB → 1024-d, configs already say 1024.)
14. Legacy configs `qmlp_dmlm_emlm_robust04_*_l1_0.0_{0.0,0.00001,0.0001,0.001}` and `qmlp_dmlm_emlm_codec_*_l1_0.0_0.00001` (despite its name it is a Robust04 config) point entity files at the authors' cluster (`/projects/0/guse0488/...`) and are **not** the paper configs; the paper's DyVo(REL, LaQue) rows are the `qmlp_dmlm_emlm_laque_*` configs.

## Data formats
Confirmed by reading `lsr/utils/dataset_utils.py`, the dataset classes and `BERTCandidate`, and by loading `DyVo/smoke_data/` through them (`smoke_data/check_readers.py` → `READERS OK`):
- **Collection** (local path): `doc_id<TAB>text` per line; `line.strip().split("\t")` must give exactly 2 fields (no tabs inside text). `irds:<name>` uses `ir_datasets` + `doc_fields` joined by spaces.
- **Queries** (local path): `qid<TAB>text`; `text_fields` is ignored for local files. `irds:` uses `query_field` (CODEC `query`, TREC `description`).
- **Qrels** (local path): JSON `{qid: {doc_id: relevance_int}}` (ids stringified).
- **Teacher scores** `ce_score_dict` (`*.json`): `{qid: {doc_id: score}}`; only queries present here are used for training; negatives are sampled from these doc ids. Scores must be ≥ 0 because of `normalize: "std"` (E1).
- **Entity candidates**: JSONL `{"id": <text id>, "entities": [int, ...]}` (extra fields ignored). Entity ids are **row indices** into the entity embedding table (vocab id = 30522 + row).
- `TripletIDDistilDataset.__getitem__` → `{"query_id","query_text","doc_id":[pos,neg],"doc_text":[pos,neg],"score":[s_pos,s_neg]}`; 40% of the time the "positive" is a random teacher-scored doc (`neg_as_pos`).
- `PredictionDataset(num_documents=-1)` → all docs; `num_documents>0` (the eval sets) → only docs appearing in qrels.

## LLMs and API keys
From prompt §3b (to be confirmed in S5.0):
- Paper LLMs: **Mixtral-8x7B-Instruct-v0.1** (open weights, Apache-2.0; synthetic CODEC training queries via Prompt A.1 and query entity candidates via Prompt A.2) and **GPT-4** (proprietary, paid OpenAI API; query entity candidates via Prompt A.2).
- All LLM outputs are shipped in `lsr42/dyvo_data`: `codec/codec_mistral_topics.tsv` + qrels (training queries), `{codec,robust04,wapo}/queries_entities_{gpt4,mistral,bm25,laque}.jsonl`, `codec/queries_test_train_entities_human.jsonl`, `queries_test_train_entities_rel.jsonl`.
- ⇒ **No API key or local LLM is needed to reproduce the paper.**
- The official configs point `query_ent_path` only at REL files → LLM rows need a merged file (S5.1).

### S5.0 — confirmed against `lsr42/dyvo_data` (listing via HF API; only the 16 small query-entity files downloaded, 2.9 MB, into `DyVo/dyvo_data/`)
Repo total **94.69 GB, 48 files** (full listing with byte sizes in `logs/hf_listing.tsv`). Prompt §3b confirmed: Mixtral CODEC training queries (`codec/codec_mistral_topics.tsv`, `codec_mistral_qrels.json`), GPT-4/Mixtral/BM25/LaQue query entities for all three datasets, CODEC human entities, REL train+test entities, `knowledge_base/entity_list.json` (129 MB), MonoT5-3B scores. `dyvo_init/` also ships a tokenizer.

Query-entity candidate files (stats over the **test** query ids; "OOV drop" = 1 − avg #ids / avg #names):

| dataset | source | #queries in file | coverage | test ids missing | avg #names returned | avg #entity ids (test) | OOV drop | queries w/ 0 entities | entity_names encoding | max entity id |
|---|---|---|---|---|---|---|---|---|---|---|
| codec | rel | 5723 | train+test (5681 train) | 0 | – | 4.7 | – | 0 | – | 5151601 |
| codec | gpt4 | 42 | test-only | 0 | 18.3 | 14.2 | 22.5% | 0 | json-string | 5103405 |
| codec | mistral | 42 | test-only | 0 | 18.9 | 10.7 | 43.6% | 0 | list | 3290300 |
| codec | bm25 | 42 | test-only | 0 | – | 20.0 | – | 0 | – | 5266067 |
| codec | laque | 42 | test-only | 0 | – | 20.0 | – | 0 | – | 5272772 |
| codec | human | 5723 | train+test (5681 train) | 0 | – | 18.0 | – | 0 | – | 5222459 |
| robust04 | rel | 10250 | train+test (10000 train) | 0 | – | 6.0 | – | 0 | – | 5268916 |
| robust04 | gpt4 | 250 | test-only | 0 | 18.6 | 33.8 | -81.9% | 0 | json-string | 234639 |
| robust04 | mistral | 250 | test-only | 0 | 21.0 | 10.7 | 49.2% | 2 | json-string | 5133166 |
| robust04 | bm25 | 250 | test-only | 0 | – | 20.0 | – | 0 | – | 5275610 |
| robust04 | laque | 250 | test-only | 0 | – | 20.0 | – | 0 | – | 5266595 |
| wapo | rel | 10048 | train+test (9998 train) | 0 | – | 19.1 | – | 0 | – | 5277084 |
| wapo | gpt4 | 50 | test-only | 0 | 18.2 | 13.0 | 28.5% | 0 | list | 4984768 |
| wapo | mistral | 50 | test-only | 0 | 19.0 | 9.1 | 52.4% | 0 | json-string | 3269849 |
| wapo | bm25 | 50 | test-only | 0 | – | 20.0 | – | 0 | – | 5259710 |
| wapo | laque | 50 | test-only | 0 | – | 20.0 | – | 0 | – | 5277304 |

- codec: 42 test query ids (from the LLM files), all present in the REL file: True; REL file has 5723 ids (5681 training ids).
- robust04: 250 test query ids (from the LLM files), all present in the REL file: True; REL file has 10250 ids (10000 training ids).
- wapo: 50 test query ids (from the LLM files), all present in the REL file: True; REL file has 10048 ids (9998 training ids).
- codec human file: same 5723 ids as REL; training ids identical to REL: 5681/5681; test ids differing from REL: 42/42 → **it is already a merged file (REL train + human test)**.

Findings:
1. **All LLM/BM25/LaQue files are test-only** (42 CODEC, 250 Robust04, 50 Core18 queries), no test query is missing; REL files cover train+test. So the LLM rows need a merged file (S5.1). Confirms §3b.
2. **`codec/queries_test_train_entities_human.jsonl` is already a merged file**: identical to REL on all 5681 training ids, human entities on all 42 test ids. It can be used directly as `query_ent_path` (merge_candidates with `--source human` reproduces it).
3. `entity_names` encoding is inconsistent: a JSON-encoded string in some files, a real JSON list in others (codec/mistral, wapo/gpt4). Parsers must accept both (`generate_entities_llm.parse_entity_list` does).
4. LLM OOV drop rates: GPT-4 22.5% (CODEC), 28.5% (Core18); Mixtral 43.6% / 49.2% / 52.4% (CODEC / Robust04 / Core18). Two Robust04 Mixtral queries end with 0 entities.
5. ⚠️ **`robust04/queries_entities_gpt4.jsonl` looks misaligned with the entity table**: every row has duplicate ids (54 ids for 23 names on query 301; 13.8 unique ids on average vs 18.6 names), all ids < 234,640 (other files go up to ~5.27M), **0%** overlap with the REL ids of the same queries (other LLM files: 7–36%), and none of the 4 queries naming "United States" contain id 0 (every other file: all do). It very likely uses a different id space (e.g. an intermediate vocabulary). Using it as-is would score the wrong entity embeddings. **Big-PC action:** rebuild ids from its `entity_names` with `python -m lsr.entity.generate_entities_llm --backend remap --input dyvo_data/robust04/queries_entities_gpt4.jsonl --entity-list dyvo_data/knowledge_base/entity_list.json --output dyvo_data/robust04/queries_entities_gpt4_remapped.jsonl` (no LLM call) and compare; `02_check_data.py` flags the file. The paper's Robust04 GPT-4 numbers (54.39 nDCG@10) can only be trusted to reproduce after this check.
6. Entity id 0 = "United States" in the correctly aligned files (as stated in §3b).

## Deviations from paper
- (code-level) Eq. 4 λ placement: code multiplies λ_ent **before** ReLU/log1p (`log1p(relu(λ·e·h))`); paper writes `λ·log1p(relu(e·h))`. We keep the code's version (it is what produced the paper's numbers). For λ>0 both are monotone in `e·h` and zero at the same places; magnitudes differ.
- (code-level) Eq. 3: code applies `log1p(relu(·))` to the MLP output; paper's Eq. 3 shows the raw linear score.

## Deferred to big PC
(filled as tasks complete)

## Sanity log
- S5.4: `lsr/entity/generate_entities_llm.py` — Prompt A.2 verbatim; backends mock / remap (no LLM) / openai / ollama / vllm (OpenAI-compatible HTTP) / anthropic (official SDK, default `claude-opus-5`, optional dep); keys from env only; resumable; authors' output format. `entity_list.json` shape is a **guess** (JSON list, index = id, per `bert_candidates.py`'s commented `self.entity_list[eid]`); the loader also accepts {id: name} / {name: id} and fails loudly otherwise. No real API call made.
- S5.5: `pytest tests/test_llm_entities.py` → 24 passed, including: the parser recovers exactly the `entity_names` of all 6 authors' GPT-4/Mixtral files (both the JSON-string and real-list encodings).
- S4.3: `bash scripts/verify.sh` → exit 0, `VERIFY PASSED (total 47s)`: guard blocks DNS, 130 .py compile, 73/73 configs compose, 53 entity_dim checks, 59 tests, overfit PASS, both smoke runs write run files. Sanity numbers identical across runs (deterministic).
- S5.1: `python -m lsr.entity.merge_candidates --dataset <ds> --source <src>` for all 13 (dataset, source) pairs → line counts equal REL (5723 / 10250 / 10048), 0 fallbacks; `--source human` output is **identical** to the authors' `queries_test_train_entities_human.jsonl` (independent confirmation of the merge logic).
- S5.2: 28 new configs (Table 2: 13, `…_entw_0.05_{gpt4,mistral,bm25,laquecand,human}.yaml`; Table 3: 15, `qmlp_dmlm_emlm_{bert,dpr,jds,wiki2vec,blink}_*_gpt4.yaml`; LaQue+GPT-4 shared with Table 2). `bash scripts/check_configs.sh` → `configs composed: 73 passed, 0 failed` (43 official + 28 + 2 smoke). E3: `# @package _global_` must stay on line 1.
- S5.3: `pytest tests/test_merge_candidates.py` → 8 passed (synthetic + real small files).
- S4.1: `python smoke_data/build_data.py` → `docs=20 train_queries=10 test_queries=5 docs_without_entities=5 queries_without_entities=2`; `python smoke_data/check_readers.py` → `READERS OK`.
- S4.2: `python -m lsr.train +experiment=smoke_dyvo` (20 steps, CPU, ~6 s) → exit 0, checkpoint + `model/` saved, `test_run.trec` 100 lines, `test_result.json` has nDCG@10/nDCG@20/R@1000 (SANITY ONLY, meaningless), train log shows `query #entities 1.8`, `doc #entities 3.0`. `smoke_lsrw` → exit 0, `query #entities 0.0`. (E2: pinned wandb 0.16.6.)
- S3.1: `python sanity/overfit_one_batch.py` (Adam lr 1e-3, 100 steps, softmax teacher, threshold final ≤ 0.5×initial) → `[SANITY ONLY] loss 0.40172 -> 0.00028 (ratio 0.001)`, non-zero entity weights 15/15 → `OVERFIT PASS`. With `--normalize std` (the configs' setting): 0.58127 → 0.00017, PASS.
- S3.2 collapse probe (300 steps each, logs in `DyVo/sanity/logs/probe_*.json`), SANITY ONLY:
  | run | loss start → end | live entity weights start → end | full collapse? |
  |---|---|---|---|
  | λ=1.0 frozen, softmax | 0.256 → 0.006 | 15 → 12 (dips to 13 by step 50) | no |
  | λ=0.05 trainable, softmax | 0.402 → 0.00007 | 15 → 15 (λ drifts to ≈0.066) | no |
  | λ=1.0 frozen, std | 0.0001 → 0.0000 | 15 → 15 | no |
  | λ=0.05 trainable, std | 0.581 → 0.00004 | 15 → 15 (λ → ≈0.073) | no |
  Honest reading: at toy scale (1 batch, random 32-d model, random entity table) the paper's collapse (App. A.2: all entity weights driven negative and killed by ReLU) does **not** reproduce. λ=1.0 frozen is the only setting where entity weights start dying (15→12), which matches the direction of A.2, but it is weak evidence. The real test is the first ~thousands of steps on the big PC (watch the `query #entities` / `doc #entities` logs).
- S2.3: `python -m pytest tests -q` → `27 passed in 1.37s` (10 test files; E1 fixed).
- S2.1: `python smoke_assets/build_assets.py` → `vocab_size == 30522`, `built smoke assets in 1.7s, total 13.5 MB`.
- S2.2: `bash scripts/check_configs.sh` → initially 32/43 (11 missing `trec_robust04_fold1`); after adding it → `configs composed: 43 passed, 0 failed`.
- S0/S1: `python -c "import lsr.models, lsr.losses, lsr.datasets, lsr.trainer"` → ok after fix #1.
