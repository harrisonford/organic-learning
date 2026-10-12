#!/usr/bin/env bash
# Second organ (parts + cross-fixation prediction + belief-change motor), unattended.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> out/pipeline2.log; }
log "developing the parts bank (once)"
ORGAN=2 MOTOR=center .venv/bin/python experiments/organ_life.py 10 /tmp/organ2_dev.jsonl >> out/pipeline2.log 2>&1
log "lives: organ 2, motor bayes/random/center seed 0, bayes seed 1; 50000 images each"
ORGAN=2 MOTOR=bayes SEED=0 .venv/bin/python experiments/organ_life.py 50000 out/organ2_bayes_s0.jsonl > out/organ2_bayes_s0.log 2>&1 &
ORGAN=2 MOTOR=bayes SEED=1 .venv/bin/python experiments/organ_life.py 50000 out/organ2_bayes_s1.jsonl > out/organ2_bayes_s1.log 2>&1 &
ORGAN=2 MOTOR=random SEED=0 .venv/bin/python experiments/organ_life.py 50000 out/organ2_random_s0.jsonl > out/organ2_random_s0.log 2>&1 &
ORGAN=2 MOTOR=center SEED=0 .venv/bin/python experiments/organ_life.py 50000 out/organ2_center_s0.jsonl > out/organ2_center_s0.log 2>&1 &
wait
log "lives: done"
for r in bayes_s0 bayes_s1 random_s0 center_s0; do
  echo "$r: $(tail -n 1 out/organ2_$r.log)" >> out/pipeline2.log
  .venv/bin/python experiments/organ_view.py out/organ2_$r.pkl organ2_$r >> out/pipeline2.log 2>&1
done
log "pipeline finished"
