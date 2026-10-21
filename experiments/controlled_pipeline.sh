#!/usr/bin/env bash
# Controlled comparison: fresh lives in one frozen world (2026-10-21: COCO val2017
# + VQA + bAbI + CIFAR-100 + that day's text world), identical except for the
# visual cortex, stopped at the same age. One life at a time: in this richer world
# a life grows ~7 GB by 80k moments. Resumable. Then builds the COCO train2017
# thumbnails (memory-heavy, so only once the lives are done).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FREEZE=1 PER_KIND=60 EVERY=20000
AGE=${AGE:-100000}
# a frozen copy of the adults' questions, so every life here lives in the same world
if [ ! -d data/world_ctl ]; then mkdir -p data/world_ctl && cp data/world/coco_qa.jsonl data/world/text_qa.jsonl data/world_ctl/; fi
export WORLD_DIR=data/world_ctl
run() {  # name, seed, extra env
  out=out/ctl_$1
  done_n=0
  if [ -f $out.pkl ]; then done_n=$(.venv/bin/python -c "import pickle,sys; sys.path.insert(0,'.'); print(pickle.load(open('$out.pkl','rb')).age)"); fi
  left=$(( AGE - done_n ))
  [ $left -le 0 ] && return
  if [ -f $out.pkl ]; then R="RESUME=$out.pkl"; else R=""; fi
  env $R $3 SEED=$2 .venv/bin/python experiments/world_life.py $left $out.jsonl >> $out.log 2>&1
}
for seed in 0 1; do
  run cortex_s$seed $seed CORTEX=data/visual_cortex.pkl
  run plain_s$seed $seed X=0
done
echo "done $(date)" > out/ctl_done
while pgrep -x curl > /dev/null; do sleep 60; done
unset WORLD_DIR
WORKERS=8 .venv/bin/python experiments/world/coco_thumbs.py ~/datasets/train2017.zip >> out/coco_train_build.log 2>&1 && \
.venv/bin/python -c "
import sys, collections; sys.path.insert(0, '.')
from organic.world import build_coco_train_items
items = build_coco_train_items(); print(len(items), collections.Counter(k for *_, k in items))" >> out/coco_train_build.log 2>&1 && echo built >> out/coco_train_build.log
