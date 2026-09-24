# NOTES

## Environment
- Build machine: Apple M4 (arm64), macOS 26.6.2, 16 GB RAM, ~14 GB free disk, **no CUDA GPU** (Apple Metal only; not used).
- Python: 3.10.20 in `.venv/` (activate with `source .venv/bin/activate`). CPU-only torch 2.2.2.
- Key pins (full list in `requirements-lock.txt`): transformers 4.40.2, tokenizers 0.19.1, accelerate 0.30.1, hydra-core 1.3.2, omegaconf 2.3.1, datasets 2.19.1, ir-datasets 0.6.3, ir-measures 0.4.3, numpy 1.26.4, faiss-cpu 1.15.1, safetensors 0.8.0, wandb 0.30.0, pytest 9.1.1.
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
5. `loss.normalize: "std"` (39 of 42 configs using it) is not a recognized name: anything other than `"softmax"` falls into `F.normalize(labels, dim=1)` (L2 normalization of teacher scores, not a probability distribution). `KLDivLoss` then compares a log-prob with a non-normalized target. Kept for reproduction.
6. `DynamicDataCollator.pad_one_group` truncates **queries with `d_max_length`** (512), not `q_max_length` (50). `q_max_length` is effectively unused by the DyVo collator. Kept (matches authors' runs); the LSR-w `DataCollator` does use `q_max_length` → a small asymmetry between LSR-w and DyVo.
7. Eval writes representations to `/scratch-shared/$USER/lsr-entities/...` (authors' cluster path) — cannot be created on most machines. Patched to honor `LSR_SCRATCH_DIR` env var, default unchanged (see Deviations/edits).
8. Eval uses `torch.autocast(device_type='cuda', dtype=float16)` — on CPU this only warns and is disabled.
9. `BertEntityEmbedding` hardcodes 30522 while the collator uses `tokenizer.vocab_size` → tiny tokenizer must have exactly 30522 entries.
10. `per_device_train_batch_size` is 128 in base `config.yaml`; the paper says batch 16 for fine-tuning. The experiment YAMLs **do not override it** → verify on big PC (could OOM at 512 tokens × 128 × 2 docs). Recorded as a big-PC check; `03_train.sh` passes `training_arguments.per_device_train_batch_size=16` to match the paper (see Deviations).
11. Base `training_arguments.dataloader_num_workers: 16`, `report_to: wandb`: smoke configs override to 0 / none.
12. Robust04 configs read `dyvo_data/robust04/docs.tsv`, not in the HF data repo (see Deferred).

## Data formats
(filled in S4.1)

## LLMs and API keys
From prompt §3b (to be confirmed in S5.0):
- Paper LLMs: **Mixtral-8x7B-Instruct-v0.1** (open weights, Apache-2.0; synthetic CODEC training queries via Prompt A.1 and query entity candidates via Prompt A.2) and **GPT-4** (proprietary, paid OpenAI API; query entity candidates via Prompt A.2).
- All LLM outputs are shipped in `lsr42/dyvo_data`: `codec/codec_mistral_topics.tsv` + qrels (training queries), `{codec,robust04,wapo}/queries_entities_{gpt4,mistral,bm25,laque}.jsonl`, `codec/queries_test_train_entities_human.jsonl`, `queries_test_train_entities_rel.jsonl`.
- ⇒ **No API key or local LLM is needed to reproduce the paper.**
- The official configs point `query_ent_path` only at REL files → LLM rows need a merged file (S5.1).

## Deviations from paper
- (code-level) Eq. 4 λ placement: code multiplies λ_ent **before** ReLU/log1p (`log1p(relu(λ·e·h))`); paper writes `λ·log1p(relu(e·h))`. We keep the code's version (it is what produced the paper's numbers). For λ>0 both are monotone in `e·h` and zero at the same places; magnitudes differ.
- (code-level) Eq. 3: code applies `log1p(relu(·))` to the MLP output; paper's Eq. 3 shows the raw linear score.

## Deferred to big PC
(filled as tasks complete)

## Sanity log
- S0/S1: `python -c "import lsr.models, lsr.losses, lsr.datasets, lsr.trainer"` → ok after fix #1.
