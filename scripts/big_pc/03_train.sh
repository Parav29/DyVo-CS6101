#!/usr/bin/env bash
# Big-PC step 3: train + evaluate ONE experiment with the official entry point, under nohup, logging to logs/<name>.log.
#   python -m lsr.train +experiment=<name> training_arguments.fp16=True training_arguments.per_device_train_batch_size=16
# Refuses to start if the same experiment is already running; skips if it already finished (unless --force).
# Usage: bash scripts/big_pc/03_train.sh <experiment-name> [--dry-run] [--wait] [--force] [-- extra hydra overrides]
#   --wait   run in the foreground-equivalent (start under nohup, then wait for it; used by run_priority.sh)
# Env: BATCH_SIZE (default 16 = paper; the YAMLs inherit 128 from config.yaml, see NOTES #10),
#      LSR_SCRATCH_DIR (eval representations; default <root>/scratch), WANDB_MODE (default offline).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY=0; WAIT=0; FORCE=0; NAME=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --wait) WAIT=1; shift ;;
    --force) FORCE=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) NAME="$1"; shift ;;
  esac
done
if [[ -z "$NAME" ]]; then
  if [[ $DRY -eq 1 ]]; then NAME="qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05"
  else echo "usage: $0 <experiment-name> [--dry-run] [--wait] [--force]" >&2; exit 2; fi
fi
CFG="$ROOT/DyVo/lsr/configs/experiment/$NAME.yaml"
[[ -f "$CFG" ]] || { echo "no such experiment config: $CFG" >&2; exit 2; }

LOGDIR="$ROOT/logs"; LOG="$LOGDIR/$NAME.log"; PIDF="$LOGDIR/$NAME.pid"
export LSR_SCRATCH_DIR="${LSR_SCRATCH_DIR:-$ROOT/scratch}"
export WANDB_MODE="${WANDB_MODE:-offline}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PY="${PYTHON:-python}"
log() { echo "[03_train $(date +%H:%M:%S)] $*"; }

# Already finished? (HFTrainer.predict writes test_result.json)
if compgen -G "$ROOT/DyVo/outputs/$NAME/*/test_result.json" >/dev/null && [[ $FORCE -eq 0 ]]; then
  log "already finished: $(ls "$ROOT"/DyVo/outputs/"$NAME"/*/test_result.json | head -1) (use --force to rerun)"
  exit 0
fi
# Already running?
if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "refusing: $NAME is already running (pid $(cat "$PIDF"), log $LOG)" >&2
  exit 3
fi
if pgrep -f "lsr.train +experiment=$NAME( |$)" >/dev/null 2>&1; then
  echo "refusing: a process for +experiment=$NAME is already running" >&2
  exit 3
fi

CMD=("$PY" -m lsr.train "+experiment=$NAME" training_arguments.fp16=True
     "training_arguments.per_device_train_batch_size=$BATCH_SIZE" ${EXTRA[@]+"${EXTRA[@]}"})
if [[ $DRY -eq 1 ]]; then
  echo "[dry-run] cd $ROOT/DyVo && LSR_SCRATCH_DIR=$LSR_SCRATCH_DIR WANDB_MODE=$WANDB_MODE nohup ${CMD[*]} > $LOG 2>&1 &"
  echo "[dry-run] pid -> $PIDF; outputs -> DyVo/outputs/$NAME/<date>/ (model/, test_run.trec, test_result.json)"
  exit 0
fi
mkdir -p "$LOGDIR" "$LSR_SCRATCH_DIR"
cd "$ROOT/DyVo"
log "starting $NAME (batch $BATCH_SIZE, fp16) -> $LOG"
nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDF"
log "pid $PID. Follow with: tail -f $LOG"
if [[ $WAIT -eq 1 ]]; then
  set +e; wait "$PID"; RC=$?; set -e
  rm -f "$PIDF"
  log "$NAME exited with $RC"
  exit $RC
fi
