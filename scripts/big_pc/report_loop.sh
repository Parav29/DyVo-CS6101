#!/usr/bin/env bash
# DYVO-REPRO: regenerate repro_codec/REPORT.md every 20 min until 3 Oct (works with no Claude session attached)
cd /home/24b4515/Dyvo_cs6101; source .venv-gpu/bin/activate
while [[ $(date +%s) -lt $(date -d '2026-10-03 12:00' +%s) ]]; do
  python scripts/big_pc/codec_report.py > logs/report_loop.last 2>&1
  sleep 1200
done
