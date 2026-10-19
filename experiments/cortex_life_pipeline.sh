#!/usr/bin/env bash
# Very long lives with the visual cortex (resumable: rerun to continue from the saved brain).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() {  # seed, total moments
  out=out/cortex_s$1
  done_n=0
  if [ -f $out.pkl ]; then done_n=$(.venv/bin/python -c "import pickle,sys; sys.path.insert(0,'.'); print(pickle.load(open('$out.pkl','rb')).age)"); fi
  left=$(( $2 - done_n ))
  [ $left -le 0 ] && return
  if [ -f $out.pkl ]; then R="RESUME=$out.pkl"; else R=""; fi
  env $R CORTEX=data/visual_cortex.pkl SEED=$1 .venv/bin/python experiments/world_life.py $left $out.jsonl >> $out.log 2>&1
}
run 0 1000000 &
run 1 500000 &
wait
echo "done $(date)" > out/cortex_life_done
