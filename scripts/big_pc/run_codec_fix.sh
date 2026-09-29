#!/usr/bin/env bash
# CODEC repair lane (runs alongside run_codec_phase2.sh; see repro_codec/REPORT.md):
#   1. DyVo(20k) re-tested from its saved model with REL candidates   [E_course20k_eval_rel]   (its own test crashed: stale scratch dir)
#   2. DyVo(20k) + GPT-4 / Mixtral candidates                         [E_course20k_eval_{gpt4,mistral}] (phase 2 skipped them)
#   3. DyVo init checkpoint, retried                                   [E_init]                 (CUDA OOM next to other jobs)
# Eval batch 8 instead of 64 so each job fits next to the two 100k runs (scores do not depend on it).
# Usage: nohup setsid bash scripts/big_pc/run_codec_fix.sh [--dry-run] > logs/run_codec_fix.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
NEED_MB="${NEED_MB:-7500}"; NEED_RAM_GB="${NEED_RAM_GB:-45}"; POLL_S="${POLL_S:-120}"; EVAL_BS="${EVAL_BS:-8}"
E=qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05
OUT="$ROOT/DyVo/outputs"
log() { echo "[run_codec_fix $(date '+%F %T')] $*"; }
pick_gpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
  awk -F', *' -v m="$NEED_MB" '($2-$3)>=m {print ($2-$3), $1}' | sort -rn | head -1 | awk '{print $2}'; }
ram_ok() { (( $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) >= NEED_RAM_GB )); }
finished() { compgen -G "$OUT/$1/*/test_result.json" >/dev/null; }
run_exp() {
  local exp="$1" gpu
  if [[ $DRY -eq 1 ]]; then BATCH_SIZE=4 bash "$ROOT/scripts/big_pc/03_train.sh" "$exp" --dry-run -- "training_arguments.per_device_eval_batch_size=$EVAL_BS"; return 0; fi
  if finished "$exp"; then log "skip (finished): $exp"; return 0; fi
  until gpu="$(pick_gpu)" && [[ -n "$gpu" ]] && ram_ok; do
    log "waiting ($exp): GPU >= ${NEED_MB} MiB free, RAM >= ${NEED_RAM_GB} GB: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';') RAM avail $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GB"
    sleep "$POLL_S"
  done
  log "starting $exp on GPU $gpu"
  set +e; CUDA_VISIBLE_DEVICES="$gpu" BATCH_SIZE=4 bash "$ROOT/scripts/big_pc/03_train.sh" "$exp" --wait -- "training_arguments.per_device_eval_batch_size=$EVAL_BS"; local rc=$?; set -e
  log "$exp exited with $rc"
}
[[ $DRY -eq 1 ]] || { d="$(ls -d "$OUT/${E}_course20k"/*/model | head -1)"; ln -sfn "$(basename "$(dirname "$d")")" "$OUT/${E}_course20k/final"; log "final -> $(dirname "$d")"; }
source "$ROOT/.venv-gpu/bin/activate"
for exp in "${E}_course20k_eval_rel" "${E}_course20k_eval_gpt4" "${E}_course20k_eval_mistral" "${E}_init"; do run_exp "$exp"; done
[[ $DRY -eq 1 ]] && { echo "[dry-run] then: 04_collect_results.py and codec_report.py"; exit 0; }
python "$ROOT/scripts/big_pc/04_collect_results.py" || true
python "$ROOT/scripts/big_pc/codec_report.py"
log "all done"
