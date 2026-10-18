#!/usr/bin/env bash
# Design 2 + contrast reflex + adaptation, with sleep cleanup, 80,000 moments.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for seed in 0 1; do
  DESIGN=2 SEED=$seed BRAIN='{"reflex": "contrast", "adapt": true, "sleep_cleanup": true}' .venv/bin/python experiments/yarbus.py 80000 out/sleep_d2_s${seed}.jsonl > out/sleep_d2_s${seed}.log 2>&1 &
done
wait
echo "done $(date)" > out/sleep_done
