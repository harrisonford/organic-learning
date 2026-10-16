#!/usr/bin/env bash
# Longer lives for the adaptation brains: continue from age 20,000 to 80,000, unchanged.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for seed in 0 1; do
  RESUME=out/d1_adapt_s${seed}.pkl SEED=$seed .venv/bin/python experiments/yarbus.py 60000 out/long_d1_adapt_s${seed}.jsonl > out/long_d1_adapt_s${seed}.log 2>&1 &
  RESUME=out/d2_contrast_adapt_s${seed}.pkl SEED=$seed DESIGN=2 .venv/bin/python experiments/yarbus.py 60000 out/long_d2_adapt_s${seed}.jsonl > out/long_d2_adapt_s${seed}.log 2>&1 &
done
wait
echo "done $(date)" > out/yarbus5_done
