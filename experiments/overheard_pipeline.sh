#!/bin/bash
# Overheard speech, controlled: two small lives (8 neurons, 2,000 clusters, the same seed and world),
# without and with 3 overheard sentences per moment. BB -> W1 -> W2. Results: out/overheard_o{0,3}.jsonl
cd "$(dirname "$0")/.."
for O in 3 0; do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 SPEECH=1 MOTOR_UNITS=300 MAX_OUTPUTS=1024 NEURONS=8 MAX_CLUSTERS=2000 OVERHEARD=$O \
  MAX_GB=2.4 EVERY=5000 STAGE_BUDGET=100000 TOTAL=${TOTAL:-200000} CORTEX=data/visual_cortex.pkl CORTEX_GRID=4 CORTEX_LAYERS=0,1,2 SEED=0 \
  nice -n 10 .venv/bin/python experiments/curriculum_life.py out/overheard_o$O.jsonl > out/overheard_o$O.log 2>&1
done
