#!/usr/bin/env bash
# Input adaptation (design 1 and 2) and the contrast reflex back in design 2.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for seed in 0 1; do
  SEED=$seed BRAIN='{"adapt": true}' .venv/bin/python experiments/yarbus.py 20000 out/d1_adapt_s${seed}.jsonl > out/d1_adapt_s${seed}.log 2>&1 &
  DESIGN=2 SEED=$seed BRAIN='{"reflex": "contrast"}' .venv/bin/python experiments/yarbus.py 20000 out/d2_contrast_s${seed}.jsonl > out/d2_contrast_s${seed}.log 2>&1 &
  DESIGN=2 SEED=$seed BRAIN='{"reflex": "contrast", "adapt": true}' .venv/bin/python experiments/yarbus.py 20000 out/d2_contrast_adapt_s${seed}.jsonl > out/d2_contrast_adapt_s${seed}.log 2>&1 &
done
wait
echo "done $(date)" > out/yarbus4_done
