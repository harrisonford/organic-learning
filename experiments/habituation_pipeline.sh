#!/usr/bin/env bash
# Is the ear's fixed habituation rule needed, or does the orchestrator's own input
# adaptation (each input neuron subtracts its slow average) do the same job?
# Two fresh lives in the frozen world (data/world_ctl), identical except for the rule.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FREEZE=1 PER_KIND=60 EVERY=20000 WORLD_DIR=data/world_ctl SEED=0
.venv/bin/python experiments/world_life.py 100000 out/hab_on_s0.jsonl > out/hab_on_s0.log 2>&1
EAR_HABITUATION=0 .venv/bin/python experiments/world_life.py 100000 out/hab_off_s0.jsonl > out/hab_off_s0.log 2>&1
echo "done $(date)" > out/hab_done
