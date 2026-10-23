#!/usr/bin/env bash
# Run (or resume) the current long life; used by the organic-life systemd user service
# so a reboot does not end it. Settings of the life live here.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_GB=11 MAX_CLUSTERS=24000 MAX_OUTPUTS=4096 EVERY=10000 PER_KIND=40 CORTEX=data/visual_cortex.pkl SEED=0
OUT=out/full_s0 TOTAL=1000000
age=0
[ -f $OUT.pkl ] && age=$(.venv/bin/python -c "import pickle,sys; sys.path.insert(0,'.'); print(pickle.load(open('$OUT.pkl','rb')).age)")
left=$(( TOTAL - age ))
[ $left -le 0 ] && exit 0
[ -f $OUT.pkl ] && export RESUME=$OUT.pkl
exec .venv/bin/python experiments/world_life.py $left $OUT.jsonl >> $OUT.log 2>&1
