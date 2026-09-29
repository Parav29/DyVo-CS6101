#!/usr/bin/env bash
# Big-PC: the recommended run order. Runs experiments one after another (each via 03_train.sh --wait), skipping
# finished ones, and collects results after each group. No LLM is called anywhere in this script.
#   priority1  CODEC: LSR-w vs DyVo(REL, LaQue), L1=1e-5
#   priority2  Robust04 + Core 2018: LSR-w vs DyVo(REL, LaQue), L1=1e-5
#   priority3  L1 sweep 1e-3, 1e-4 (LSR-w + DyVo REL, all datasets)                         -> Table 1
#   priority4  Table 2: LLM/BM25/LaQue/Human query candidates (authors' files, merged with REL training entities)
#   priority5  Table 3: entity-embedding ablation with GPT-4 candidates (needs all 6 embedding tables, ~92 GB)
# Usage: bash scripts/big_pc/run_priority.sh [--dry-run] [--groups priority1,priority2,...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY=0; GROUP_SEL="priority1,priority2,priority3,priority4,priority5"  # not GROUPS: that is a bash builtin array
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --groups) GROUP_SEL="$2"; shift 2 ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
PY="${PYTHON:-python}"
BIG="$ROOT/scripts/big_pc"
D=""; [[ $DRY -eq 1 ]] && D="--dry-run"
log() { echo; echo "[run_priority $(date +%H:%M:%S)] $*"; }
run() { if [[ $DRY -eq 1 ]]; then echo "[dry-run] $*"; else echo "+ $*"; "$@"; fi; }

prepare_candidates() {
  log "building merged query-candidate files (REL for training ids + source for test ids)"
  cd "$ROOT/DyVo"
  if [[ -f dyvo_data/knowledge_base/entity_list.json || $DRY -eq 1 ]]; then
    # robust04 GPT-4 file looks like a different id space (NOTES S5.0 #5): rebuild ids from its entity_names.
    run "$PY" -m lsr.entity.generate_entities_llm --backend remap --no-resume \
        --input dyvo_data/robust04/queries_entities_gpt4.jsonl --entity-list dyvo_data/knowledge_base/entity_list.json \
        --output dyvo_data/robust04/queries_entities_gpt4_remapped.jsonl
  else
    echo "WARNING: entity_list.json missing -> robust04 GPT-4 ids cannot be re-mapped; using the authors' file as-is"
  fi
  for ds in codec robust04 wapo; do
    for src in gpt4 mistral bm25 laque; do
      extra=()
      if [[ "$ds/$src" == "robust04/gpt4" && ( -f dyvo_data/robust04/queries_entities_gpt4_remapped.jsonl || $DRY -eq 1 ) ]]; then
        extra=(--source-file dyvo_data/robust04/queries_entities_gpt4_remapped.jsonl)
      fi
      run "$PY" -m lsr.entity.merge_candidates --dataset "$ds" --source "$src" ${extra[@]+"${extra[@]}"}
    done
  done
  run "$PY" -m lsr.entity.merge_candidates --dataset codec --source human
  cd "$ROOT"
}

IFS=',' read -r -a GROUP_LIST <<< "$GROUP_SEL"
ALL_EXPS=()
NEED_CANDS=0
for g in "${GROUP_LIST[@]}"; do
  EXPS="$("$PY" "$BIG/experiments.py" --group "$g")"   # command substitution: an unknown group aborts under set -e
  for exp in $EXPS; do ALL_EXPS+=("$exp"); done
  [[ "$g" == "priority4" || "$g" == "priority5" ]] && NEED_CANDS=1
done
[[ $NEED_CANDS -eq 1 ]] && prepare_candidates   # merged candidate files must exist before the data check

log "data check for ${#ALL_EXPS[@]} experiment(s) (fix every FAIL before training)"
EXP_CSV="$(IFS=,; echo "${ALL_EXPS[*]}")"
run "$PY" "$BIG/02_check_data.py" --experiments "$EXP_CSV" || { echo "02_check_data failed; aborting"; exit 1; }

for g in "${GROUP_LIST[@]}"; do
  log "group $g"
  for exp in $("$PY" "$BIG/experiments.py" --group "$g"); do
    bash "$BIG/03_train.sh" "$exp" --wait $D
  done
  run "$PY" "$BIG/04_collect_results.py" $D
  [[ "$g" == "priority1" ]] && run "$PY" "$BIG/codec_report.py"   # CODEC-only reproduction report
done
log "all requested groups done. Results: results/results.csv"
