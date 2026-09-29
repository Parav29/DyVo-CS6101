#!/usr/bin/env bash
# Blocks until something in the CODEC runs needs attention, prints it, and exits (for a supervising session).
# Events: a new "exited with" / "skip" line in an orchestrator log; a Traceback / OOM / Killed in a running
# experiment log; an experiment log without progress for STALL_MIN minutes; an orchestrator that died.
# Usage: bash scripts/big_pc/watch_events.sh [--dry-run]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ "${1:-}" == "--dry-run" ]] && { echo "[dry-run] would watch logs/run_codec_*.log and running experiment logs"; exit 0; }
LOGS="$ROOT/logs"; STATE="$LOGS/.watch_seen"; STALL_MIN="${STALL_MIN:-30}"; HEARTBEAT_MIN="${HEARTBEAT_MIN:-180}"; T0=$(date +%s)
touch "$STATE"
while true; do
  for o in "$LOGS"/run_codec_course.log "$LOGS"/run_codec_phase2.log "$LOGS"/run_codec_fix.log; do
    [[ -f "$o" ]] || continue
    while IFS= read -r l; do
      grep -qxF "$l" "$STATE" || { echo "$l" >> "$STATE"; echo "EVENT: $l"; ev=1; }
    done < <(grep -E "exited with|skip \(finished\)|all done|final ->" "$o")
  done
  for pidf in "$LOGS"/*.pid; do
    [[ -f "$pidf" ]] || continue
    name="$(basename "$pidf" .pid)"; log="$LOGS/$name.log"
    if grep -qE "Traceback|CUDA out of memory|Killed" "$log" 2>/dev/null && ! grep -qxF "ERR $name" "$STATE"; then
      echo "ERR $name" >> "$STATE"; echo "EVENT: error in $name: $(grep -E 'Error|out of memory|Killed' "$log" | tail -2 | cut -c1-200)"; ev=1
    fi
    age=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
    if (( age >= STALL_MIN )) && ! grep -qxF "STALL $name $(( age / STALL_MIN ))" "$STATE"; then
      echo "STALL $name $(( age / STALL_MIN ))" >> "$STATE"; echo "EVENT: $name log unchanged for ${age} min"; ev=1
    fi
  done
  for o in run_codec_course run_codec_phase2 run_codec_fix; do
    if [[ -f "$LOGS/$o.log" ]] && ! grep -q "all done" "$LOGS/$o.log" && ! pgrep -f "$o.sh" >/dev/null \
       && ! grep -qxF "DEAD $o" "$STATE"; then
      echo "DEAD $o" >> "$STATE"; echo "EVENT: orchestrator $o is not running and did not finish"; ev=1
    fi
  done
  if [[ "${ev:-0}" != 1 ]] && (( $(date +%s) - T0 >= HEARTBEAT_MIN * 60 )); then
    echo "HEARTBEAT (${HEARTBEAT_MIN} min, no events)"; ev=1
  fi
  if [[ "${ev:-0}" == 1 ]]; then   # progress summary of every running experiment
    for pidf in "$LOGS"/*.pid; do
      [[ -f "$pidf" ]] || continue; name="$(basename "$pidf" .pid)"
      echo "  $name: $(tr '\r' '\n' < "$LOGS/$name.log" | grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)"
      echo "    $(grep -oE "\{'query reg'[^}]*\}" "$LOGS/$name.log" | tail -1 | grep -oE "'(query|doc) #entities': [0-9.]+|'loss_no_reg': [0-9.na]+" | tr '\n' ' ')"
    done
    exit 0
  fi
  sleep 60
done
