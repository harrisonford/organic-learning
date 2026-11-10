#!/bin/bash
# Word cells in a life, three seeds at once (docs/AREAS.md; the ruler: 300 test items, 3 seeds per claim).
# Small brains as in the hub lives (8 neurons, 2,000 clusters), from babbling: BB -> B -> W1 -> W2 ...
# with word cells made of the ear's own units (organic/words.py), unasked comments on pictures (1 per moment),
# overheard speech (1 sentence per moment) and the local judge (reward = acceptance; the gate stays on the local grade).
# Results: out/${TAG:-words}_s{0,1,2}.jsonl
cd "$(dirname "$0")/.."
for S in ${SEEDS:-0 1 2}; do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 SPEECH=1 WORDS=${WORDS:-1} HUB=${HUB:-0} HUB_CELLS=1500 MAX_UNITS=2000 MAX_CUES=8192 OVERHEARD=1 OBSERVE=${OBSERVE:-1} JUDGE=${JUDGE:-1} \
  MOTOR_UNITS=300 MAX_OUTPUTS=1024 NEURONS=8 MAX_CLUSTERS=2000 MAX_GB=4 EVERY=5000 STAGE_BUDGET=${STAGE_BUDGET:-100000} TOTAL=${TOTAL:-400000} TEST_N=300 \
  CORTEX=data/visual_cortex.pkl CORTEX_GRID=4 CORTEX_LAYERS=0,1,2 SEED=$S \
  nice -n 10 .venv/bin/python experiments/curriculum_life.py out/${TAG:-words}_s$S.jsonl > out/${TAG:-words}_s$S.log 2>&1 &
done
wait
