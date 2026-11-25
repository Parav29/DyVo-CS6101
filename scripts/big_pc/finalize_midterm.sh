#!/usr/bin/env bash
# DYVO-REPRO: wait for the LSR-w 100k test, fill it into the write-up, commit everything, push (retrying until it works).
cd /home/24b4515/Dyvo_cs6101; source .venv-gpu/bin/activate
L=qmlp_dmlm_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_full100k
LOG=logs/finalize_midterm.log
log() { echo "[finalize $(date '+%F %T')] $*" >> $LOG; }
log "waiting for DyVo/outputs/$L/fasteval/test_result.json"
until [[ -f DyVo/outputs/$L/fasteval/test_result.json ]]; do sleep 120; done
sleep 30
python scripts/big_pc/finalize_midterm.py >> $LOG 2>&1
python scripts/big_pc/codec_report.py >> $LOG 2>&1
bash scripts/big_pc/codec_snapshot.sh >> $LOG 2>&1 || log "codec_snapshot failed (non-fatal)"
git add repro_codec scripts DyVo/lsr project_context.md >> $LOG 2>&1
git commit -q -m "CODEC repro: LSR-w 100k result, all test results (top-1000 runs), final midterm write-up

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> $LOG 2>&1 && log "committed $(git log --oneline -1)"
while [[ $(date +%s) -lt $(date -d '2026-10-03 12:00' +%s) ]]; do
  if GIT_TERMINAL_PROMPT=0 git push origin master >> $LOG 2>&1; then log "PUSHED"; exit 0; fi
  log "push failed (no GitHub credentials yet); retry in 10 min"; sleep 600
done
log "gave up pushing"
