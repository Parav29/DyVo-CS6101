#!/usr/bin/env bash
# CODEC reproduction, phase 2 (runs alongside run_codec_course.sh; see repro_codec/REPORT.md):
#   lane W: LSR-w, paper length (100k steps, batch 4 x 4)                                  [W_full100k]   starts now
#   lane E: waits until phase 1's DyVo lane is done (only one DyVo process fits in RAM), then
#           1. DyVo(20k) + GPT-4 query candidates at test time (paper headline, Table 2)   [E_course20k_eval_gpt4]
#           2. DyVo, paper length (100k steps)                                             [E_full100k]
#           3. DyVo(100k) + GPT-4 candidates                                               [E_full100k_eval_gpt4]
#           4. Mixtral candidates on both models                                           [E_*_eval_mistral]
# GPU: waits for >= NEED_MB free. RAM: DyVo jobs also wait for >= NEED_RAM_GB available (entity table = 16 GB,
# process ~38 GB). Finished runs are skipped. Ends with 04_collect_results.py and codec_report.py.
# Usage: nohup setsid bash scripts/big_pc/run_codec_phase2.sh [--dry-run] > logs/run_codec_phase2.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
NEED_MB="${NEED_MB:-9000}"; NEED_RAM_GB="${NEED_RAM_GB:-45}"; POLL_S="${POLL_S:-120}"
W=qmlp_dmlm_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001
E=qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05
OUT="$ROOT/DyVo/outputs"
log() { echo "[run_codec_phase2 $(date '+%F %T')] $*"; }

pick_gpu() {  # GPU with the most free memory, if >= NEED_MB
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
    awk -F', *' -v m="$NEED_MB" '($2-$3)>=m {print ($2-$3), $1}' | sort -rn | head -1 | awk '{print $2}'
}
ram_ok() { (( $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) >= NEED_RAM_GB )); }
finished() { compgen -G "$OUT/$1/*/test_result.json" >/dev/null; }

run_exp() {  # $1 = experiment, $2 = 1 if it needs the DyVo RAM budget
  local exp="$1" big="${2:-0}" gpu
  if [[ $DRY -eq 1 ]]; then BATCH_SIZE=4 bash "$ROOT/scripts/big_pc/03_train.sh" "$exp" --dry-run; return 0; fi
  if finished "$exp"; then log "skip (finished): $exp"; return 0; fi
  until gpu="$(pick_gpu)" && [[ -n "$gpu" ]] && { [[ $big -eq 0 ]] || ram_ok; }; do
    log "waiting ($exp): GPU >= ${NEED_MB} MiB free$([[ $big -eq 1 ]] && echo ", RAM >= ${NEED_RAM_GB} GB available"): $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';') RAM avail $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GB"
    sleep "$POLL_S"
  done
  log "starting $exp on GPU $gpu"
  set +e; CUDA_VISIBLE_DEVICES="$gpu" BATCH_SIZE=4 bash "$ROOT/scripts/big_pc/03_train.sh" "$exp" --wait; local rc=$?; set -e
  log "$exp exited with $rc"
  return $rc
}

link_final() {  # outputs/<exp>/final -> the dated dir holding its finished model (used by the *_eval_* configs)
  [[ $DRY -eq 1 ]] && { echo "[dry-run] ln -sfn <date> $OUT/$1/final"; return 0; }
  local d; d="$(dirname "$(ls -t "$OUT"/"$1"/*/test_result.json | head -1)")"
  ln -sfn "$(basename "$d")" "$OUT/$1/final"; log "final -> $d"
}

lane_w() { run_exp "${W}_full100k" 0 || true; }

lane_e() {
  if [[ $DRY -eq 0 ]]; then
    until grep -qE "(${E}_init exited with|skip \(finished\): ${E}_init)" "$ROOT/logs/run_codec_course.log" 2>/dev/null; do
      log "waiting for phase 1's DyVo lane (${E}_init) to finish"; sleep 600
    done
  fi
  if [[ $DRY -eq 1 ]] || finished "${E}_course20k"; then
    link_final "${E}_course20k"
    run_exp "${E}_course20k_eval_gpt4" 1 || true
  else
    log "DyVo course20k has no test result; skipping its GPT-4/Mixtral evals"
  fi
  if run_exp "${E}_full100k" 1; then
    link_final "${E}_full100k"
    run_exp "${E}_full100k_eval_gpt4" 1 || true
  fi
  if [[ $DRY -eq 1 ]] || finished "${E}_course20k"; then run_exp "${E}_course20k_eval_mistral" 1 || true; fi
  if [[ $DRY -eq 1 ]] || finished "${E}_full100k"; then run_exp "${E}_full100k_eval_mistral" 1 || true; fi
}

source "$ROOT/.venv-gpu/bin/activate"
lane_w & PW=$!
lane_e & PE=$!
wait $PW $PE || true
[[ $DRY -eq 1 ]] && { echo "[dry-run] then: 04_collect_results.py and codec_report.py"; exit 0; }
python "$ROOT/scripts/big_pc/04_collect_results.py" || true
python "$ROOT/scripts/big_pc/codec_report.py"
log "all done"
