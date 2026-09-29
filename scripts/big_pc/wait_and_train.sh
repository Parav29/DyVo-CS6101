#!/usr/bin/env bash
# Big-PC (shared machine): wait until the corpus is present AND a GPU is free, then launch run_priority.sh on it.
# "Free" = at least MIN_FREE_MB free memory and at most MAX_UTIL % utilization for STABLE consecutive polls
# (so we do not grab a GPU during a short gap between someone else's jobs).
# Usage: nohup setsid bash scripts/big_pc/wait_and_train.sh [--dry-run] [--groups priority1] > logs/wait_and_train.log 2>&1 &
# Env: MIN_FREE_MB (default 20000), MAX_UTIL (default 20), STABLE (default 3), POLL_S (default 300),
#      CORPUS (default ~/.ir_datasets/codec/v1/comets_documents.jsonl; the file the first group needs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROUP_SEL="priority1"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --groups) GROUP_SEL="$2"; shift 2 ;;
    -h|--help) sed -n 2,7p "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
MIN_FREE_MB="${MIN_FREE_MB:-20000}"; MAX_UTIL="${MAX_UTIL:-20}"; STABLE="${STABLE:-3}"; POLL_S="${POLL_S:-300}"
CORPUS="${CORPUS:-$HOME/.ir_datasets/codec/v1/comets_documents.jsonl}"
log() { echo "[wait_and_train $(date '+%F %T')] $*"; }

# Prints the index of the free GPU with the most free memory, or nothing.
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', *' -v m="$MIN_FREE_MB" -v u="$MAX_UTIL" '($2-$3)>=m && $4<=u {print ($2-$3), $1}' |
    sort -rn | head -1 | awk '{print $2}'
}

log "waiting: groups=$GROUP_SEL, need corpus $CORPUS and a GPU with >=${MIN_FREE_MB} MiB free, <=${MAX_UTIL}% util, ${STABLE} polls in a row (every ${POLL_S}s)"
if [[ $DRY -eq 1 ]]; then
  [[ -s "$CORPUS" ]] && log "corpus present" || log "corpus missing: $CORPUS"
  gpu="$(pick_gpu || true)"; log "free GPU now: ${gpu:-none}"
  log "[dry-run] would run: run_priority.sh --groups $GROUP_SEL on the first GPU free for $STABLE polls"
  exit 0
fi
streak=0; last=""
while true; do
  if [[ ! -s "$CORPUS" ]]; then
    log "corpus missing: $CORPUS"; streak=0
  else
    gpu="$(pick_gpu || true)"
    if [[ -n "$gpu" && ( "$gpu" == "$last" || $streak -eq 0 ) ]]; then
      streak=$((streak + 1)); last="$gpu"
      log "GPU $gpu free ($streak/$STABLE)"
      [[ $streak -ge $STABLE ]] && break
    else
      streak=0; last=""
      log "no free GPU: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
    fi
  fi
  sleep "$POLL_S"
done

export CUDA_VISIBLE_DEVICES="$gpu"
log "starting run_priority.sh --groups $GROUP_SEL on GPU $gpu"
source "$ROOT/.venv-gpu/bin/activate"
set +e; bash "$ROOT/scripts/big_pc/run_priority.sh" --groups "$GROUP_SEL"; RC=$?; set -e
log "run_priority.sh exited with $RC"
exit $RC
