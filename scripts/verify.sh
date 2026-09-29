#!/usr/bin/env bash
# Offline verification gate for the DyVo build phase (Levels 0-4). Fully offline, CPU only, ~2-3 min.
# Usage: bash scripts/verify.sh            (exit 0 = all green)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

# --- Resource rules: offline + CPU, enforced ---------------------------------------------------------------
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=""
export WANDB__SERVICE_WAIT=300   # wandb service start timeout (default 30 s is too short under heavy load)
export DYVO_BLOCK_NETWORK=1                       # scripts/offline_guard/sitecustomize.py: outbound sockets raise
export PYTHONPATH="$ROOT/scripts/offline_guard${PYTHONPATH:+:$PYTHONPATH}"
export LSR_SCRATCH_DIR="$ROOT/DyVo/smoke_outputs/scratch"
export PYTHONHASHSEED=0
PY=python
T0=$(date +%s)
step() { echo; echo "=== [$(( $(date +%s) - T0 ))s] $*"; }

step "Network guard active"
$PY - <<'EOF'
import socket
try:
    socket.getaddrinfo("huggingface.co", 443)
    raise SystemExit("offline guard NOT active")
except RuntimeError as e:
    print("ok:", e)
EOF

cd "$ROOT/DyVo"
if [[ ! -f smoke_assets/dyvo_init/doc_encoder/model.safetensors ]]; then step "Building tiny assets"; $PY smoke_assets/build_assets.py; fi
if [[ ! -f smoke_data/docs.tsv ]]; then step "Building smoke data"; $PY smoke_data/build_data.py; fi

# --- Level 0: static -----------------------------------------------------------------------------------------
step "L0: py_compile every .py"
n_py=$(find "$ROOT/DyVo" "$ROOT/scripts" -name "*.py" -not -path "*/outputs/*" -not -path "*/smoke_outputs/*" | wc -l | tr -d ' ')
find "$ROOT/DyVo" "$ROOT/scripts" -name "*.py" -not -path "*/outputs/*" -not -path "*/smoke_outputs/*" -print0 \
  | xargs -0 $PY -m py_compile
echo "py_compile ok: $n_py files"

step "L0: bash -n every .sh"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$ROOT/scripts" -name "*.sh" -print0)
echo "bash -n ok"

step "L0: import core modules"
$PY - <<'EOF'
import importlib, pkgutil
import lsr.models, lsr.losses, lsr.datasets, lsr.trainer
n = 0
for pkg in (lsr.models, lsr.losses, lsr.datasets, lsr.trainer):
    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{m.name}"); n += 1
import lsr.entity.bert_candidates, lsr.entity.merge_candidates, lsr.entity.generate_entities_llm  # noqa
print(f"imported {n} submodules of lsr.models/losses/datasets/trainer + entity modules")
EOF

step "L0: compose every experiment config (--cfg job)"
bash "$ROOT/scripts/check_configs.sh"

step "L0: entity_dim matches each entity table's width"
$PY "$ROOT/scripts/check_entity_dims.py"

# --- Level 1 + 2: unit tests + behavioral invariants ----------------------------------------------------------
step "L1: pytest"
$PY -m pytest tests -q -p no:cacheprovider

step "L2: overfit one batch"
$PY sanity/overfit_one_batch.py --log-json smoke_outputs/overfit.json | tail -3

# --- Level 3: end-to-end smoke through the official entry point -----------------------------------------------
for exp in smoke_dyvo smoke_lsrw; do
  step "L3: python -m lsr.train +experiment=$exp"
  rm -rf "outputs/$exp"
  mkdir -p "$ROOT/logs"
  $PY -m lsr.train "+experiment=$exp" > "$ROOT/logs/$exp.log" 2>&1 || { tail -30 "$ROOT/logs/$exp.log"; exit 1; }
  run=$(ls outputs/$exp/*/test_run.trec)
  res=$(ls outputs/$exp/*/test_result.json)
  [[ -s "$run" ]] || { echo "empty run file $run"; exit 1; }
  ls -d outputs/$exp/*/model/query_encoder outputs/$exp/*/model/doc_encoder >/dev/null
  $PY - "$res" "$run" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("test_nDCG@10", "test_nDCG@20", "test_R@1000"):
    assert k in m, k
n = sum(1 for _ in open(sys.argv[2]))
print(f"[SANITY ONLY - meaningless numbers] nDCG@10={m['test_nDCG@10']:.4f} nDCG@20={m['test_nDCG@20']:.4f} "
      f"R@1000={m['test_R@1000']:.4f}; run file lines={n}")
EOF
done

# --- Level 4: big-PC scripts dry-run --------------------------------------------------------------------------
step "L4: big-PC scripts --dry-run"
shopt -s nullglob
for f in "$ROOT"/scripts/big_pc/*; do
  case "$f" in
    *.sh) bash -n "$f"; bash "$f" --dry-run > "$ROOT/logs/dryrun_$(basename "$f").log" 2>&1 || { tail -20 "$ROOT/logs/dryrun_$(basename "$f").log"; echo "DRY-RUN FAIL: $f"; exit 1; } ;;
    *.py) $PY -m py_compile "$f"; $PY "$f" --dry-run > "$ROOT/logs/dryrun_$(basename "$f").log" 2>&1 || { tail -20 "$ROOT/logs/dryrun_$(basename "$f").log"; echo "DRY-RUN FAIL: $f"; exit 1; } ;;
    *) continue ;;
  esac
  echo "dry-run ok: $(basename "$f")"
done

step "VERIFY PASSED (total $(( $(date +%s) - T0 ))s)"
