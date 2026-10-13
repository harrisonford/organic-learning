#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for loops in both reflex goal; do for seed in 0 1; do
  DESIGN=2 LOOPS=$loops SEED=$seed .venv/bin/python experiments/yarbus.py 20000 out/yarbus2_${loops}_s${seed}.jsonl > out/yarbus2_${loops}_s${seed}.log 2>&1 &
done; done
wait
echo "done $(date)" > out/yarbus2_done
