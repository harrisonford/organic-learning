#!/bin/bash
# The association hub, controlled: two small lives (8 neurons, 2,000 clusters, seed 0, the wider world,
# 1 overheard sentence per moment), with and without the hub. BB -> W1 -> W2 ... Results: out/hub_h{1,0}.jsonl
cd "$(dirname "$0")/.."
for H in ${HS:-1 0}; do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 SPEECH=1 HUB=$H HUB_CELLS=1500 OVERHEARD=1 MOTOR_UNITS=300 MAX_OUTPUTS=1024 NEURONS=8 MAX_CLUSTERS=2000 \
  MAX_GB=3.5 EVERY=5000 STAGE_BUDGET=100000 TOTAL=${TOTAL:-250000} CORTEX=data/visual_cortex.pkl CORTEX_GRID=4 CORTEX_LAYERS=0,1,2 SEED=0 \
  nice -n 10 .venv/bin/python experiments/curriculum_life.py out/hub_${TAG:-h}$H.jsonl > out/hub_${TAG:-h}$H.log 2>&1
done
