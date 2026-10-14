#!/usr/bin/env bash
# Longer lives (continue to 80k) + design-2 diagnosis one change at a time.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for seed in 0 1; do
  RESUME=out/yarbus_both_s${seed}.pkl SEED=$seed .venv/bin/python experiments/yarbus.py 60000 out/long1_both_s${seed}.jsonl > out/long1_both_s${seed}.log 2>&1 &
  RESUME=out/yarbus2_both_s${seed}.pkl SEED=$seed DESIGN=2 .venv/bin/python experiments/yarbus.py 60000 out/long2_both_s${seed}.jsonl > out/long2_both_s${seed}.log 2>&1 &
  DESIGN=2 SEED=$seed BRAIN='{"noise": 0.0, "use_stay": false, "use_gains": false}' .venv/bin/python experiments/yarbus.py 20000 out/abl_none_s${seed}.jsonl > out/abl_none_s${seed}.log 2>&1 &
  DESIGN=2 SEED=$seed BRAIN='{"noise": 0.1, "use_stay": false, "use_gains": false}' .venv/bin/python experiments/yarbus.py 20000 out/abl_noise_s${seed}.jsonl > out/abl_noise_s${seed}.log 2>&1 &
  DESIGN=2 SEED=$seed BRAIN='{"noise": 0.0, "use_stay": true, "use_gains": false}' .venv/bin/python experiments/yarbus.py 20000 out/abl_stay_s${seed}.jsonl > out/abl_stay_s${seed}.log 2>&1 &
  DESIGN=2 SEED=$seed BRAIN='{"noise": 0.0, "use_stay": false, "use_gains": true}' .venv/bin/python experiments/yarbus.py 20000 out/abl_gains_s${seed}.jsonl > out/abl_gains_s${seed}.log 2>&1 &
done
wait
echo "done $(date)" > out/yarbus3_done
