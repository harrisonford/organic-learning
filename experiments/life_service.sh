#!/usr/bin/env bash
# Run (or resume) the current long life; used by the organic-life systemd user service
# so a reboot does not end it. Current life: kid-a training -- the speaking brain through
# the curriculum, cumulatively from the simplest stage; every promotion leaves a release
# candidate in out/levels/ (served by ~/kid-a).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SPEECH=1 MAX_GB=11 MAX_CLUSTERS=24000 EVERY=5000 STAGE_BUDGET=300000 TOTAL=5000000 CORTEX=data/visual_cortex.pkl SEED=0
OUT=out/kid_s0
grep -q '"verdict": "' $OUT.jsonl 2>/dev/null && exit 0  # the life already has its verdict
[ -f $OUT.pkl ] && export RESUME=$OUT.pkl
exec .venv/bin/python experiments/curriculum_life.py $OUT.jsonl >> $OUT.log 2>&1
