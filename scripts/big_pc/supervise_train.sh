#!/usr/bin/env bash
# DYVO-REPRO: keep a 100k run alive on the shared server until it finishes, then test it with fast_eval.py.
#   supervise_train.sh <experiment> [fast_eval args...]
# Each attempt resumes from the newest checkpoint-* of that experiment (any dated dir), on the GPU with the most free
# memory, with the largest per-device batch that fits (gradient accumulation keeps the effective batch at 16).
# Crashes (CUDA OOM from other users' jobs, host OOM kills) -> wait and retry from the newest checkpoint.
set -u
ROOT=/home/24b4515/Dyvo_cs6101
EXP=$1; shift
LOG=$ROOT/logs/supervise_$EXP.log
cd $ROOT; source .venv-gpu/bin/activate
log() { echo "[supervise $(date '+%F %T')] $*" >> "$LOG"; }

latest_ckpt() {
  ls -d $ROOT/DyVo/outputs/$EXP/*/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2
}
final_model() { ls -d $ROOT/DyVo/outputs/$EXP/*/model/doc_encoder 2>/dev/null | head -1 | xargs -r dirname; }
pick_gpu() {  # prints "<gpu> <free MiB>" for the GPU with most free memory
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | tr -d ' ' | tr , ' '
}

for attempt in $(seq 1 200); do
  M=$(final_model)
  [[ -n "$M" ]] && break
  read GPU FREE < <(pick_gpu)
  if   (( FREE >= 9000 )); then BS=4; GA=4
  elif (( FREE >= 5500 )); then BS=2; GA=8
  elif (( FREE >= 3500 )); then BS=1; GA=16
  else log "only ${FREE} MiB free on best GPU $GPU; waiting"; sleep 300; continue; fi
  CK=$(latest_ckpt)
  RES=()
  [[ -n "$CK" ]] && RES=(resume_from_checkpoint=$CK)
  log "attempt $attempt: GPU $GPU (${FREE} MiB free) batch ${BS}x${GA} resume ${CK:-none}"
  CUDA_VISIBLE_DEVICES=$GPU LSR_SKIP_PREDICT=1 BATCH_SIZE=$BS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/big_pc/03_train.sh "$EXP" --wait --force -- training_arguments.gradient_accumulation_steps=$GA \
    training_arguments.dataloader_num_workers=4 training_arguments.ignore_data_skip=True training_arguments.save_steps=2500 \
    "${RES[@]}" >> "$LOG" 2>&1
  RC=$?
  cp "$ROOT/logs/$EXP.log" "$ROOT/logs/$EXP.attempt$attempt.log" 2>/dev/null
  log "attempt $attempt exited $RC; last step: $(grep -o "'epoch': [0-9.]*" $ROOT/logs/$EXP.log | tail -1)"
  M=$(final_model)
  [[ -n "$M" ]] && break
  sleep 120
done
M=$(final_model)
[[ -z "$M" ]] && { log "gave up"; exit 1; }
log "training finished -> $M; testing"
bash scripts/big_pc/run_fasteval.sh "$(pick_gpu | cut -d' ' -f1)" "fasteval_$EXP" --model-dir "$M" "$@"
log "fast_eval exit $?"
