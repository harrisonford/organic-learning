#!/bin/bash
# Cluster size (neurons per cluster) controlled test: 4 / 8 / 16 neurons, the same
# total neurons (clusters x size = 16000), the same seed, world, eye, ear and
# curriculum (BB -> W1 ...). Small brains, run one after another, niced, next to the
# main life without taking its memory. Results: out/csize_n{4,8,16}.jsonl
cd "$(dirname "$0")/.."
for N in 8 4 16; do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 SPEECH=1 MOTOR_UNITS=300 MAX_OUTPUTS=1024 NEURONS=$N MAX_CLUSTERS=$((16000 / N)) \
  MAX_GB=2.4 EVERY=5000 STAGE_BUDGET=100000 TOTAL=${TOTAL:-250000} CORTEX=data/visual_cortex.pkl CORTEX_GRID=4 CORTEX_LAYERS=0,1,2 SEED=0 \
  nice -n 10 .venv/bin/python experiments/curriculum_life.py out/csize_n$N.jsonl > out/csize_n$N.log 2>&1
done
