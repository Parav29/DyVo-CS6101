#!/usr/bin/env bash
# CODEC-only course-project reproduction (see repro_codec/REPORT.md). Two lanes in parallel (LSR-w | DyVo), each runs:
#   1. course budget training: batch 4 x 4 accumulation (= 16), 20k steps, then full-corpus test  [*_course20k]
#   2. the same model evaluated at the authors' init checkpoint, no training                     [*_init]
# Before each run it waits for a GPU with >= NEED_MB free (measured: DyVo peaks at 6.0 GB), preferring the lane's own
# GPU (LSR-w -> 0, DyVo -> 1) so the lanes spread over both cards, and pins the run to it.
# Finished runs are skipped (03_train.sh). Ends by writing results/results.csv and repro_codec/REPORT.md.
# Usage: nohup setsid bash scripts/big_pc/run_codec_course.sh [--dry-run] > logs/run_codec_course.log 2>&1 &
# Env: NEED_MB (default 9000), POLL_S (default 120)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
NEED_MB="${NEED_MB:-9000}"; POLL_S="${POLL_S:-120}"
BASE_W=qmlp_dmlm_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001
BASE_E=qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05
log() { echo "[run_codec_course $(date '+%F %T')] $*"; }

pick_gpu() {  # $1 = preferred GPU if it has >= NEED_MB free, else the GPU with the most free memory (if >= NEED_MB)
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
    awk -F', *' -v m="$NEED_MB" -v pref="$1" '($2-$3)>=m {print ($1==pref ? 1e9 : $2-$3), $1}' | sort -rn | head -1 | awk '{print $2}'
}

lane() {  # $1 = preferred GPU, rest = experiments to run in order
  local pref="$1"; shift
  for exp in "$@"; do
    if [[ $DRY -eq 1 ]]; then BATCH_SIZE=4 bash "$ROOT/scripts/big_pc/03_train.sh" "$exp" --dry-run; continue; fi
    if compgen -G "$ROOT/DyVo/outputs/$exp/*/test_result.json" >/dev/null; then log "skip (finished): $exp"; continue; fi
    until gpu="$(pick_gpu "$pref")" && [[ -n "$gpu" ]]; do
      log "waiting for a GPU with ${NEED_MB} MiB free: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
      sleep "$POLL_S"
    done
    log "starting $exp on GPU $gpu"
    set +e; CUDA_VISIBLE_DEVICES="$gpu" BATCH_SIZE=4 bash "$ROOT/scripts/big_pc/03_train.sh" "$exp" --wait; rc=$?; set -e
    log "$exp exited with $rc"
  done
}

source "$ROOT/.venv-gpu/bin/activate"
lane 0 "${BASE_W}_course20k" "${BASE_W}_init" & PID_W=$!
[[ $DRY -eq 1 ]] || sleep 60   # stagger start-up: both lanes read the corpus / entity table from NFS
lane 1 "${BASE_E}_course20k" "${BASE_E}_init" & PID_E=$!
wait $PID_W $PID_E || true
[[ $DRY -eq 1 ]] && { echo "[dry-run] then: 04_collect_results.py and codec_report.py"; exit 0; }
python "$ROOT/scripts/big_pc/04_collect_results.py" || true
python "$ROOT/scripts/big_pc/codec_report.py"
log "all done"
