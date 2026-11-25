#!/bin/bash
# Extension 2 v2 evaluation + plan step 8 (E1 recomputed on the repaired index). Runs after the delta dump.
cd /home/24b4515/Dyvo_cs6101
source .venv-gpu/bin/activate
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
until grep -q -E "EXIT CODE|Traceback" logs/ext_dump_delta_v2.log; do sleep 30; done
grep -q "EXIT CODE 0" logs/ext_dump_delta_v2.log || { echo "delta dump failed"; exit 1; }
cd scripts/ext
E1=/home/24b4515/Dyvo_cs6101/scratch/ext/e1/results_prf
step() { echo "=== $(date) $*"; "$@"; rc=$?; echo "=== rc=$rc $*"; [ $rc -eq 0 ] || exit $rc; }
step python -u e2_run.py --v2 --e1 E1occ-v2-K3=$E1/queries_E1-occ_k3.jsonl PRFscore-K5=$E1/queries_C-prf-score_k5.jsonl
step python -u e2_make_index.py --variant grounded --cap 5 --v2 --out index_grounded_v2
step python -u e1_prf.py --index index_grounded_v2 --tag _g2
step python -u e1_labels.py --index index_grounded_v2 --pool _prf_g2 --tag _prf_g2
step python -u e1_run.py --index index_grounded_v2 --pool _prf_g2 --tag _prf_g2
step python -u e1_cvk.py --tag _prf_g2
echo "ALL DONE $(date)"
