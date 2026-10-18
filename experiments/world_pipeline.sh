#!/usr/bin/env bash
# Long lives in the photo world (resumable: reruns continue from the saved brain).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for seed in 0 1; do
  if [ -f out/world_s${seed}.pkl ]; then R="RESUME=out/world_s${seed}.pkl"; else R=""; fi
  env $R SEED=$seed .venv/bin/python experiments/world_life.py 300000 out/world_s${seed}.jsonl >> out/world_s${seed}.log 2>&1 &
done
wait
echo "done $(date)" > out/world_done
