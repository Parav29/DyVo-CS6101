#!/usr/bin/env bash
# Level 0: compose every experiment config with `python -m lsr.train +experiment=<name> --cfg job`.
# This prints the composed config and exits before train() runs: nothing is loaded or downloaded.
# Usage: bash scripts/check_configs.sh [--only-real] [--pattern <glob>]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/DyVo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=""
PY="${PYTHON:-python}"

ONLY_REAL=0
PATTERN="*.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-real) ONLY_REAL=1; shift ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

pass=0; fail=0; failed=()
for f in lsr/configs/experiment/$PATTERN; do
  name="$(basename "$f" .yaml)"
  if [[ $ONLY_REAL -eq 1 && ( "$name" == smoke_* || "$name" == *_gpt4 || "$name" == *_mistral || "$name" == *_bm25 || "$name" == *_laquecand || "$name" == *_human ) ]]; then
    continue
  fi
  if out="$("$PY" -m lsr.train "+experiment=$name" --cfg job 2>&1)"; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1)); failed+=("$name")
    echo "FAIL: $name"; echo "$out" | tail -5 | sed 's/^/    /'
  fi
done
echo "configs composed: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
