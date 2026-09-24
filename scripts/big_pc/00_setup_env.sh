#!/usr/bin/env bash
# Big-PC step 0: create the Python env with the CUDA build of torch + the locked requirements, then check the GPU.
# Idempotent: reuses an existing env; re-running only re-checks.
# Usage: bash scripts/big_pc/00_setup_env.sh [--dry-run] [--venv PATH] [--cuda cu121|cu118] [--python python3.10]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY=0; VENV="$ROOT/.venv-gpu"; CUDA="cu121"; PYBIN="${PYBIN:-python3.10}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    --cuda) CUDA="$2"; shift 2 ;;
    --python) PYBIN="$2"; shift 2 ;;
    -h|--help) sed -n 2,5p "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
log() { echo "[00_setup_env $(date +%H:%M:%S)] $*"; }
run() { if [[ $DRY -eq 1 ]]; then echo "[dry-run] $*"; else log "+ $*"; "$@"; fi; }

TORCH_VERSION="2.2.2"   # same as requirements-lock.txt (CPU build on the build machine)
LOCK="$ROOT/requirements-lock.txt"
[[ -f "$LOCK" ]] || { echo "missing $LOCK" >&2; exit 1; }

if [[ -x "$VENV/bin/python" ]]; then
  log "env exists: $VENV (reusing)"
else
  command -v "$PYBIN" >/dev/null 2>&1 || [[ $DRY -eq 1 ]] || { echo "need $PYBIN (or pass --python / use conda: conda create -n dyvo python=3.10)" >&2; exit 1; }
  run "$PYBIN" -m venv "$VENV"
fi
PIP="$VENV/bin/python -m pip"
run $PIP install --upgrade pip
# CUDA torch first, then everything else from the lock file except torch/torchvision (CPU pins).
run $PIP install "torch==$TORCH_VERSION" "torchvision==0.17.2" --index-url "https://download.pytorch.org/whl/$CUDA"
FILTERED="$ROOT/logs/requirements-gpu.txt"
if [[ $DRY -eq 1 ]]; then
  echo "[dry-run] grep -vE '^(torch|torchvision)==' $LOCK > $FILTERED"
else
  mkdir -p "$ROOT/logs"; grep -vE '^(torch|torchvision)==' "$LOCK" > "$FILTERED"
fi
run $PIP install -r "$FILTERED"
run $PIP install "huggingface_hub[cli]" psutil
# Optional, only for 05_generate_entities.sh --backend anthropic:  $PIP install anthropic

CHECK='import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__); \
assert torch.cuda.is_available(), "CUDA not available: check the driver / --cuda flag"; \
print("GPU:", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30, 1), "GiB")'
run "$VENV/bin/python" -c "$CHECK"
log "done. Activate with: source $VENV/bin/activate"
