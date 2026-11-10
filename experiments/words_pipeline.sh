#!/bin/bash
# Word cells in a life, three seeds at once (docs/AREAS.md; the ruler: 300 test items, 3 seeds per claim).
# Small brains as in the hub lives (8 neurons, 2,000 clusters), from babbling: BB -> B -> W1 -> W2 ...
# with word cells made of the ear's own units (organic/words.py), unasked comments on pictures (1 per moment),
# overheard speech (1 sentence per moment) and the local judge (reward = acceptance; the gate stays on the local grade).
# Each life is a systemd user unit organic-${TAG}-s$S (it outlives the session; it resumes from its last checkpoint
# if out/${TAG}_s$S.pkl exists). Results: out/${TAG:-words}_s{0,1,2}.jsonl; list them in out/monitor/lives.json.
#     experiments/words_pipeline.sh            SEEDS="0 1 2" TAG=words
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG=${TAG:-words}
for S in ${SEEDS:-0 1 2}; do
  RES=(); [ -f "$ROOT/out/${TAG}_s$S.pkl" ] && RES=(-E RESUME=out/${TAG}_s$S.pkl)
  systemd-run --user --unit=organic-$TAG-s$S --collect -p WorkingDirectory="$ROOT" -p Nice=10 "${RES[@]}" \
    -E OMP_NUM_THREADS=3 -E OPENBLAS_NUM_THREADS=3 -E SPEECH=1 -E WORDS=1 -E HUB=0 -E MAX_UNITS=2000 -E MAX_CUES=8192 -E OVERHEARD=1 -E OBSERVE=${OBSERVE:-1} -E JUDGE=${JUDGE:-1} \
    -E MOTOR_UNITS=300 -E MAX_OUTPUTS=1024 -E NEURONS=8 -E MAX_CLUSTERS=2000 -E MAX_GB=4 -E EVERY=5000 -E STAGE_BUDGET=${STAGE_BUDGET:-100000} -E TOTAL=${TOTAL:-400000} -E TEST_N=300 \
    -E CORTEX=data/visual_cortex.pkl -E CORTEX_GRID=4 -E CORTEX_LAYERS=0,1,2 -E SEED=$S \
    -p StandardOutput=append:"$ROOT/out/${TAG}_s$S.log" -p StandardError=append:"$ROOT/out/${TAG}_s$S.log" \
    "$ROOT/.venv/bin/python" experiments/curriculum_life.py out/${TAG}_s$S.jsonl
done
