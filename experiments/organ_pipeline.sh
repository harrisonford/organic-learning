#!/usr/bin/env bash
# Runs the whole progressive-eye study unattended (survives a dropped SSH session
# when started with setsid/nohup). Logs to out/pipeline.log.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*" >> out/pipeline.log; }
mkdir -p out ~/datasets

log "download: resuming CIFAR-10"
until [ -f ~/datasets/cifar-10-batches-py/test_batch ]; do
  curl -sSL -C - -o ~/datasets/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
    && tar xzf ~/datasets/cifar-10-python.tar.gz -C ~/datasets || { log "download hiccup, retrying in 30s"; sleep 30; }
done
log "download: done"

BASELINE=1 .venv/bin/python experiments/organ_life.py >> out/pipeline.log 2>&1

log "lives: 4 motor variants, seed 0, 50000 images"
for m in settle curious random center; do
  MOTOR=$m SEED=0 .venv/bin/python experiments/organ_life.py 50000 out/organ_${m}_s0.jsonl > out/organ_${m}_s0.log 2>&1 &
done
wait
log "lives: done"
for m in settle curious random center; do
  tail -n 1 out/organ_${m}_s0.log >> out/pipeline.log
  .venv/bin/python experiments/organ_view.py out/organ_${m}_s0.pkl ${m} >> out/pipeline.log 2>&1
done
log "pipeline finished"
