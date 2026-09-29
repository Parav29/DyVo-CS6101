# DyVo: handoff to the big PC

On the build machine, the code was written, patched and checked end to end with tiny random models (see `SANITY_REPORT.md`; `bash scripts/verify.sh` is green in about 47 s). **Nothing real has been trained yet.** This guide takes you from copying the folder to a results table next to the paper's numbers.

## 1. Hardware you need
| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU | 1 × 24 GB (fp16) | 1 × A100 40/80 GB (the paper's setup) | Doc encoder: DistilBERT-MLM at 512 tokens; logits of 32 docs × 512 × 30,522 per step at batch 16 |
| CPU RAM | 32 GB (LaQue / DPR / JDS / Token-Aggr. tables are 16.2 GB each) | 64 GB; BLINK (21.6 GB) needs ≥ 48 GB | `BertEntityEmbedding` `torch.load`s the **whole** entity table into CPU RAM. The corpus text is also held in RAM by the dataset |
| Disk | ~25 GB (LaQue only, all 3 datasets: 18.1 GB of `dyvo_data` + outputs) + licensed corpora | ~120 GB (all 6 embedding tables for Table 3 ≈ 92.9 GB, full repo 94.7 GB) + corpora + checkpoints | Each run saves checkpoints every 5k steps (`save_total_limit: 2`, ~0.5 GB each) |

## 2. Licensed / external data to obtain first
The experiment configs read the collections through `ir_datasets` (`irds:`):
- **Robust04**: `disks45/nocr/trec-robust-2004`. TREC Disks 4 & 5 (minus CR) from NIST (https://trec.nist.gov/data/cd45/); place them as described at https://ir-datasets.com/disks45.html.
- **TREC Core 2018**: `wapo/v2/trec-core-2018`. Washington Post v2 from NIST (https://trec.nist.gov/data/wapost/); see https://ir-datasets.com/wapo.html.
- **CODEC**: `codec`. Public (despite ir_datasets saying "on request"): `curl -L https://huggingface.co/datasets/macavaney/codec/resolve/main/documents.jsonl.gz | gunzip > ~/.ir_datasets/codec/v1/comets_documents.jsonl` (729,824 docs; see ERRORS E6).
- **`dyvo_data/robust04/docs.tsv`**: **not needed.** This corrects the original plan: the Robust04 training config reads `irds:disks45/nocr/trec-robust-2004`, and its `docs.tsv` line is commented out. Build it only if you switch that config to the local TSV: run `lsr/preprocess/export_robust04_docs.py` on the licensed corpus. The format is `doc_id<TAB>title body`, one line per document, with no tabs or newlines inside the text.

## 3. Exact command sequence
Run each script with `--dry-run` first, then for real. All commands start from the project root.
```bash
# 0. environment: CUDA torch 2.2.2 (cu121) + the locked requirements; checks torch.cuda.is_available()
bash scripts/big_pc/00_setup_env.sh --dry-run && bash scripts/big_pc/00_setup_env.sh      # add --cuda cu118 for older drivers
source .venv-gpu/bin/activate
bash scripts/verify.sh                        # the offline gate must still be green on the big PC

# 1. data: lists lsr42/dyvo_data (dataset repo), checks disk, downloads selectively
python scripts/big_pc/01_list_and_download_data.py --dry-run
python scripts/big_pc/01_list_and_download_data.py --datasets codec,robust04,wapo --embeddings laque        # Tables 1-2 (18.1 GB)
# later, for Table 3:  --embeddings laque,wiki2vec,blink,dpr,jds,bert_agg

# 2. check everything the runs need (paths, RAM, ir_datasets, id alignment, teacher scores, entity_dim)
python scripts/big_pc/02_check_data.py --experiments priority1

# 3. first run (CODEC, the smallest): LSR-w and DyVo(REL, LaQue) at L1=1e-5
bash scripts/big_pc/03_train.sh qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05
tail -f logs/qmlp_dmlm_emlm_laque_codec_*.log

# 4. everything, in the recommended order (sequential; finished runs are skipped)
bash scripts/big_pc/run_priority.sh --dry-run
bash scripts/big_pc/run_priority.sh --groups priority1,priority2          # or all five groups
python scripts/big_pc/04_collect_results.py                                # -> results/results.csv (ours vs paper, diff)
```
Priority groups (`scripts/big_pc/experiments.py`, 46 runs in total):

| Group | What | Runs |
|---|---|---|
| priority1 | CODEC LSR-w vs DyVo (REL, LaQue), L1 = 1e-5 | 2 |
| priority2 | Robust04 + Core 2018, same pair | 4 |
| priority3 | L1 sweep 1e-3 / 1e-4, all datasets (rest of Table 1) | 12 |
| priority4 | Table 2: GPT-4, Mixtral, BM25, LaQue, Human candidates (authors' files, merged by `merge_candidates.py`) | 13 |
| priority5 | Table 3: Token Aggr., DPR, JDS, Wiki2Vec, BLINK with GPT-4 candidates (LaQue+GPT-4 is shared with Table 2) | 15 |

Settings `03_train.sh` applies, and why:
- `training_arguments.fp16=True`: this is the README's command.
- `per_device_train_batch_size=16`: the paper's batch size. The YAMLs inherit 128 from `config.yaml` (NOTES #10); override with `BATCH_SIZE=...`.
- `WANDB_MODE=offline` unless you set it.
- `LSR_SCRATCH_DIR=<root>/scratch`: evaluation representations go here.

## 4. Expected runtime (estimate, not measured)
- The paper reports 100k steps at batch 16 on one A100, with no timing given.
- Our rough estimate is 0.2–0.4 s/step at fp16 (512-token docs, MLM head over 30,522 terms), so **6–12 h of training per run**.
- On top of that, training stops every 5k steps for an eval on the qrels docs, and the final test does a brute-force score over the full corpus: CODEC 729k docs, Robust04 528k, Core18 595k. Allow 1–2 h more.
- Budget **~8–15 A100-hours per run**: about 1 day for priority1+2 and about 2–4 GPU-weeks for all 46 runs.
- Time saver (optional, not implemented): the Table 2/3 runs differ from the REL run only in the **test** query candidates, because training IDs keep REL. You could evaluate one trained checkpoint with several candidate files instead of retraining. Record that as a deviation if you do it.

## 5. What to check after the first run
1. **Did it run?** `DyVo/outputs/<exp>/<date>/` should contain `model/`, `test_run.trec` (about 10,000 lines per query) and `test_result.json`. Also run `python scripts/big_pc/04_collect_results.py`.
2. **No entity collapse (paper App. A.2).** In the log, `query #entities` and `doc #entities` must stay > 0 throughout. If they drop to 0 and stay there, the run collapsed; check that `entity_weight: 0.05` is set and λ is trainable.
3. **No NaN loss.** `loss_no_reg` must be finite. If it is NaN from step 1, the teacher scores are negative. `02_check_data.py` flags this (`normalize: "std"`, E1).
4. **Magnitude.** CODEC DyVo (REL) should land near nDCG@10 53.40 / R@1k 70.60, and LSR-w near 52.61 / 69.07. A gap of a point or two is normal variance; a gap of 10+ points means something is miswired. Check the init checkpoint loaded (no "missing keys" beyond `vocab_projector.weight`) and that the batch size is 16.
5. **Memory.** Watch RAM while the entity table loads (16.2 GB). If it OOMs, see `02_check_data.py`'s mmap hint for `lsr/models/entity_emb.py`.
6. **`entity_list.json`.** Its format was guessed on the build machine (JSON list, index = entity ID). `02_check_data.py` loads it and spot-checks that "United States" = 0. If the loader complains, adapt `load_entity_vocab()` in `lsr/entity/generate_entities_llm.py` **before** priority4/5, because the Robust04 GPT-4 re-mapping depends on it.

## 6. Do I need LLMs or API keys?
**Short answer: no. You need no paid API key and no local LLM to reproduce any number in the paper.** The authors' data repo already contains every LLM output the paper used.

### 6.1 Which LLMs the paper used
| Model | Used for | Open source? | How to run it yourself |
|---|---|---|---|
| Mixtral-8x7B-Instruct-v0.1 | (1) generating the 10k synthetic CODEC training queries (Prompt A.1); (2) generating query entity candidates (Prompt A.2) | **Yes**: open weights, Apache 2.0, free | Hugging Face `mistralai/Mixtral-8x7B-Instruct-v0.1`, or Ollama `mixtral:8x7b` (≈26 GB 4-bit; needs ≈32 GB RAM/VRAM). Ollama serves an OpenAI-compatible API at `http://localhost:11434/v1` |
| GPT-4 | Query entity candidates (Prompt A.2) | **No**: proprietary | Only through OpenAI's **paid** API (needs `OPENAI_API_KEY`) |

Every other model in the paper is open, and its expensive outputs are precomputed in the data repo: DistilBERT, the MonoT5-3B teacher, LaQue, BLINK, DPR, Wikipedia2Vec and the REL linker.

### 6.2 What the authors ship (verified on the build machine)
- Mixtral training queries for CODEC: `codec/codec_mistral_topics.tsv` + `codec/codec_mistral_qrels.json` (query IDs like `mistral_16`).
- GPT-4 query entities: `{codec,robust04,wapo}/queries_entities_gpt4.jsonl`.
- Mixtral query entities: `{codec,robust04,wapo}/queries_entities_mistral.jsonl`.
- BM25 / LaQue query entities: `queries_entities_bm25.jsonl`, `queries_entities_laque.jsonl`.
- Human entities (CODEC): `codec/queries_test_train_entities_human.jsonl`. This file is **already merged**: REL for all 5,681 training IDs, human for the 42 test IDs.
- REL entities (train + test): `queries_test_train_entities_rel.jsonl`. Name ↔ ID mapping: `knowledge_base/entity_list.json`. MonoT5-3B scores: `*_monot5_3b_scores.json`.

Measured on the downloaded files (test queries only; OOV = names the LLM returned that are not in the knowledge base):

| Dataset | Test queries | GPT-4 names → IDs (OOV drop) | Mixtral names → IDs (OOV drop) | Missing test queries |
|---|---|---|---|---|
| CODEC | 42 | 18.3 → 14.2 (22.5%) | 18.9 → 10.7 (43.6%) | 0 |
| Robust04 | 250 | 18.6 names, 13.8 unique IDs ⚠️ see below | 21.0 → 10.7 (49.2%); 2 queries end with 0 entities | 0 |
| Core 2018 | 50 | 18.2 → 13.0 (28.5%) | 19.0 → 9.1 (52.4%) | 0 |

BM25 and LaQue files: exactly 20 entities per test query. All LLM/BM25/LaQue files are **test-only**; training always uses REL.

⚠️ **`robust04/queries_entities_gpt4.jsonl` looks misaligned:**
- It shares 0% of its IDs with the REL IDs of the same queries (other files share 7–36%).
- Every row has duplicate IDs.
- All IDs are below 235k, while other files go up to 5.27M.
- "United States" never maps to ID 0.

`run_priority.sh` therefore rebuilds its IDs from the file's own `entity_names` using `entity_list.json` (`generate_entities_llm --backend remap`, no LLM involved) before merging. Compare both versions if the Robust04 GPT-4 numbers look off.

### 6.3 When you *would* need an LLM
Only to regenerate candidates, to test a different LLM, or to run DyVo on new queries. The generator is `lsr/entity/generate_entities_llm.py`, which uses the paper's Prompt A.2 verbatim. The wrapper `scripts/big_pc/05_generate_entities.sh`:
- prints a warning that the authors' files exist and that other models are a deviation,
- checks keys and servers,
- generates for the test topics, then merges with REL.

Options:
```bash
# Mixtral locally via Ollama (free; ≈26 GB download, ≈32 GB RAM/VRAM)
ollama pull mixtral:8x7b && ollama serve &
bash scripts/big_pc/05_generate_entities.sh --backend ollama --model mixtral:8x7b --dataset codec

# GPT-4 via OpenAI (paid; ~340 short calls cover all test sets: 42 CODEC + 250 Robust04 + 50 Core18)
export OPENAI_API_KEY=...        # the script refuses to run without it
bash scripts/big_pc/05_generate_entities.sh --backend openai --model gpt-4 --dataset codec

# Mixtral (or any HF model) via a vLLM OpenAI-compatible server
bash scripts/big_pc/05_generate_entities.sh --backend vllm --model mistralai/Mixtral-8x7B-Instruct-v0.1 --dataset codec

# Claude via the official anthropic SDK (pip install anthropic; ANTHROPIC_API_KEY or `ant auth login`) - a deviation
bash scripts/big_pc/05_generate_entities.sh --backend anthropic --model claude-opus-5 --dataset codec
```
A smaller open model (e.g. Llama 3.1 8B, Qwen2.5 7B/14B via Ollama) runs on much less hardware. It counts as a **deviation** and will likely score lower (the paper found Mixtral < GPT-4, with more OOV names and fewer entities).

### 6.4 Why the extra configs exist
The official configs are wired only to REL entities: every `query_ent_path`, including the Table 3 configs whose caption says GPT-4, points at `queries_test_train_entities_rel.jsonl`. There is also no LLM-calling code in the repo. The LLM rows become reproducible through two additions:
- `lsr/entity/merge_candidates.py` builds `queries_test_train_entities_<source>.jsonl`: REL for training IDs, the chosen source for test IDs.
- 28 new configs (`*_entw_0.05_{gpt4,mistral,bm25,laquecand,human}.yaml`; Table 3: `qmlp_dmlm_emlm_{bert,dpr,jds,wiki2vec,blink}_*_gpt4.yaml`) are exact copies of the official ones with only `query_ent_path` changed.

## 7. Known deviations and fixes you should know about (full list in NOTES.md)
- Six upstream bugs were fixed (see `SANITY_REPORT.md`). The two you would hit first:
  - the missing loss imports, which break every run;
  - the Wiki2Vec `entity_dim`, which crashes Table 3 Wiki2Vec runs.
- λ_ent sits inside the ReLU in the code (`log1p(relu(λ·e·h))`), not outside it as in Eq. 4. The code's version is kept.
- `loss.normalize: "std"` is really L2 normalization of the teacher scores. It is kept, and it requires non-negative scores.
- The DyVo collator truncates queries at 512 (`d_max_length`), while the LSR-w collator uses 50. Kept as in upstream.
- The batch size is set to 16 (paper) on the command line, because the YAMLs would otherwise use 128.
