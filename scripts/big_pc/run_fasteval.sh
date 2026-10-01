#!/usr/bin/env bash
# DYVO-REPRO: detached wrapper for fast_eval.py. Usage: run_fasteval.sh <gpu> <logname> <fast_eval args...>
# Retries (resuming from its score checkpoint) if killed, e.g. by CUDA OOM from other users' jobs.
cd /home/24b4515/Dyvo_cs6101
source .venv-gpu/bin/activate
export CUDA_VISIBLE_DEVICES=$1 WANDB_MODE=offline HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/$2.log; shift 2
for try in $(seq 1 30); do
  echo "=== try $try $(date)" >> $LOG
  python scripts/big_pc/fast_eval.py "$@" >> $LOG 2>&1 && { echo "=== DONE $(date)" >> $LOG; exit 0; }
  echo "=== failed (exit $?) $(date); retry in 120s" >> $LOG; sleep 120
done
echo "=== GAVE UP $(date)" >> $LOG; exit 1
