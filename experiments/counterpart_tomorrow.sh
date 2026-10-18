#!/usr/bin/env bash
cd "$(dirname "$0")/.."
today=$(date +%F)
while [ "$(date +%F)" = "$today" ]; do sleep 300; done
sleep 600
.venv/bin/python experiments/counterpart.py > data/world/counterpart.log 2>&1
