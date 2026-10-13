#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for loops in both reflex goal; do for seed in 0 1; do
  LOOPS=$loops SEED=$seed .venv/bin/python experiments/yarbus.py 20000 out/yarbus_${loops}_s${seed}.jsonl > out/yarbus_${loops}_s${seed}.log 2>&1 &
done; done
wait
echo "done $(date)" > out/yarbus_done
