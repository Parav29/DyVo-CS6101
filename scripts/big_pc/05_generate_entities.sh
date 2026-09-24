#!/usr/bin/env bash
# Big-PC step 5 (OPTIONAL — not needed to reproduce the paper): generate query entity candidates with an LLM using
# the paper's Prompt A.2 (lsr/entity/generate_entities_llm.py), then merge them with REL training entities.
# Usage: bash scripts/big_pc/05_generate_entities.sh --backend ollama|openai|anthropic|vllm [--model NAME]
#                                                    [--dataset codec|robust04|wapo] [--dry-run]
# Input queries = the test topics (id + text) taken from the authors' GPT-4 file, so no ir_datasets access is needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY=0; BACKEND=""; MODEL=""; DATASET="codec"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    -h|--help) sed -n 2,6p "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$BACKEND" && $DRY -eq 1 ]] && BACKEND="ollama"
case "$BACKEND" in ollama|openai|anthropic|vllm) ;; *) echo "--backend must be ollama|openai|anthropic|vllm" >&2; exit 2 ;; esac
case "$DATASET" in codec|robust04|wapo) ;; *) echo "--dataset must be codec|robust04|wapo" >&2; exit 2 ;; esac
case "$BACKEND" in
  ollama) MODEL="${MODEL:-mixtral:8x7b}" ;;
  openai) MODEL="${MODEL:-gpt-4}" ;;
  anthropic) MODEL="${MODEL:-claude-opus-5}" ;;
  vllm) MODEL="${MODEL:-mistralai/Mixtral-8x7B-Instruct-v0.1}" ;;
esac
PY="${PYTHON:-python}"
DATA="$ROOT/DyVo/dyvo_data"
SLUG="$(echo "${BACKEND}_${MODEL}" | tr -c 'A-Za-z0-9._\n-' '_')"
OUT="$DATA/$DATASET/queries_entities_${SLUG}.jsonl"
MERGED="$DATA/$DATASET/queries_test_train_entities_${SLUG}.jsonl"

cat <<EOF
============================================================================================================
NOTE: the authors already ship GPT-4 and Mixtral candidates (dyvo_data/<dataset>/queries_entities_{gpt4,mistral}.jsonl).
You do NOT need this script to reproduce the paper. Using a model other than GPT-4 / Mixtral-8x7B-Instruct-v0.1
is a deviation from the paper and must be reported as such.
backend=$BACKEND model=$MODEL dataset=$DATASET
============================================================================================================
EOF

if [[ $DRY -eq 0 ]]; then
  case "$BACKEND" in
    openai) [[ -n "${OPENAI_API_KEY:-}" ]] || { echo "refusing: OPENAI_API_KEY is not set" >&2; exit 3; } ;;
    ollama)
      BASE="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"; HOST="${BASE%/v1}"
      curl -sf "$HOST/api/tags" >/dev/null || { echo "ollama server not responding at $HOST (run: ollama serve)" >&2; exit 3; }
      curl -sf "$HOST/api/tags" | grep -q "\"${MODEL}\"" || curl -sf "$HOST/api/tags" | grep -q "\"name\":\"${MODEL}" \
        || { echo "model $MODEL not pulled (run: ollama pull $MODEL  # ~26 GB for mixtral:8x7b)" >&2; exit 3; } ;;
    anthropic) "$PY" -c "import anthropic" 2>/dev/null || { echo "pip install anthropic" >&2; exit 3; } ;;
    vllm) curl -sf "${VLLM_BASE_URL:-http://localhost:8000/v1}/models" >/dev/null || { echo "vLLM server not responding" >&2; exit 3; } ;;
  esac
  [[ -f "$DATA/knowledge_base/entity_list.json" ]] || { echo "missing knowledge_base/entity_list.json (01_list_and_download_data.py)" >&2; exit 3; }
fi

GEN=("$PY" -m lsr.entity.generate_entities_llm --backend "$BACKEND" --model "$MODEL"
     --queries "dyvo_data/$DATASET/queries_entities_gpt4.jsonl" --entity-list dyvo_data/knowledge_base/entity_list.json
     --output "dyvo_data/$DATASET/queries_entities_${SLUG}.jsonl")
MERGE=("$PY" -m lsr.entity.merge_candidates --dataset "$DATASET" --source gpt4
       --source-file "dyvo_data/$DATASET/queries_entities_${SLUG}.jsonl"
       --output "dyvo_data/$DATASET/queries_test_train_entities_${SLUG}.jsonl")
if [[ $DRY -eq 1 ]]; then
  echo "[dry-run] cd $ROOT/DyVo && ${GEN[*]}"
  echo "[dry-run] cd $ROOT/DyVo && ${MERGE[*]}"
  echo "[dry-run] then train/eval with: bash scripts/big_pc/03_train.sh <a *_gpt4 config> -- entity_candidate_retriever.query_ent_path=$MERGED"
  exit 0
fi
cd "$ROOT/DyVo"
"${GEN[@]}"          # resumable: re-running skips queries already in $OUT
"${MERGE[@]}"
echo "done: $OUT and $MERGED"
echo "evaluate with e.g.: bash scripts/big_pc/03_train.sh qmlp_dmlm_emlm_laque_${DATASET}_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05_gpt4 -- entity_candidate_retriever.query_ent_path=dyvo_data/$DATASET/queries_test_train_entities_${SLUG}.jsonl"
