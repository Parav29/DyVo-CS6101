#!/usr/bin/env bash
# CODEC-only reproduction: record resolved configs, dependency versions, hardware and code revision into repro_codec/.
# Safe to re-run (read-only except for repro_codec/). Does not touch the GPU.
# Usage: bash scripts/big_pc/codec_snapshot.sh [--dry-run]
set -euo pipefail
if [[ "${1:-}" == "--dry-run" ]]; then echo "[dry-run] would write resolved configs, env, hardware and data provenance to repro_codec/"; exit 0; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/repro_codec"; mkdir -p "$OUT/resolved_configs" "$OUT/env"
source "$ROOT/.venv-gpu/bin/activate"
EXPS=(qmlp_dmlm_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001
      qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05)

# 1. resolved Hydra configs, with exactly the overrides 03_train.sh passes
cd "$ROOT/DyVo"
for e in "${EXPS[@]}"; do
  HF_HUB_OFFLINE=1 python -m lsr.train "+experiment=$e" training_arguments.fp16=True \
    training_arguments.per_device_train_batch_size="${BATCH_SIZE:-16}" --cfg job > "$OUT/resolved_configs/$e.yaml"
  echo "resolved: $e ($(wc -l < "$OUT/resolved_configs/$e.yaml") lines)"
done

# 2. code revision (upstream commit + local DYVO-REPRO patches)
{ echo "upstream: https://github.com/thongnt99/DyVo"; git log -1 --format='commit %H (%cd)'
  echo "local modifications (git status --short):"; git status --short; } > "$OUT/env/code_revision.txt" 2>&1
git diff > "$OUT/env/local_patches.diff" 2>/dev/null || true

# 3. dependency versions
python -m pip freeze > "$OUT/env/pip_freeze.txt"
python - > "$OUT/env/versions.txt" <<'EOF'
import sys, torch, transformers, tokenizers, ir_datasets, ir_measures, numpy
print("python", sys.version.split()[0])
for m in (torch, transformers, tokenizers, ir_datasets, ir_measures, numpy): print(m.__name__, m.__version__)
print("cuda (torch build)", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
EOF

# 4. hardware
{ echo "host: $(hostname)"; echo "date: $(date -Is)"; uname -a
  echo; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket'
  echo; free -g
  echo; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  echo; echo "GPU load at snapshot time (shared machine):"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; } > "$OUT/env/hardware.txt" 2>&1

# 5. data provenance
{ echo "CODEC corpus: https://huggingface.co/datasets/macavaney/codec documents.jsonl.gz sha256 c567900f432622637677060b5d24808bc0e4b6b38806b045d8815a0bedadab90"
  echo "  -> ~/.ir_datasets/codec/v1/comets_documents.jsonl ($(wc -l < ~/.ir_datasets/codec/v1/comets_documents.jsonl) docs)"
  echo "authors' data: huggingface.co/datasets/lsr42/dyvo_data (files used, sha256):"
  cd "$ROOT/DyVo"
  sha256sum dyvo_data/codec/codec_mistral_topics.tsv dyvo_data/codec/codec_mistral_qrels.json \
    dyvo_data/codec/codec_monot5_3b_scores.json dyvo_data/codec/queries_test_train_entities_rel.jsonl \
    dyvo_data/codec/docs_entities.jsonl dyvo_data/dyvo_init/query_encoder/model.safetensors \
    dyvo_data/dyvo_init/doc_encoder/model.safetensors
  ls -l dyvo_data/knowledge_base/entity_embs_laque.pt; } > "$OUT/env/data_provenance.txt" 2>&1
echo "snapshot written to $OUT"
