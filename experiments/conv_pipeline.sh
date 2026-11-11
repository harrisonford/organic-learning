#!/bin/bash
# Lives after docs/CONVERSATION.md (2026-11-11): the world with echo games worded by the adults and social routines
# (WORLD=2), stages passed on the items a parrot of the last word gets wrong (PARROT=1), no hand-set recency in the
# copy route (the ear's novelty instead), and one of two ways of speaking:
#     ARM=routes  the orchestrator still chooses copy / own habits / recall (NOVEL_COPY=1)
#     ARM=stream  no choice: outer and inner speech prime the mouth together (ONE_STREAM=1)
# Otherwise as experiments/words_pipeline.sh (word cells, observation, overheard speech, the local judge).
# Each life is a systemd user unit organic-$TAG-s$S; it resumes from out/${TAG}_s$S.pkl if that exists.
#     ARM=stream TAG=stream SEEDS="0" experiments/conv_pipeline.sh       EXTRA="-E IMAGERY=1" adds options
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARM=${ARM:-stream}
TAG=${TAG:-$ARM}
[ "$ARM" = stream ] && SPEAK=(-E ONE_STREAM=1) || SPEAK=(-E NOVEL_COPY=1)
for S in ${SEEDS:-0 1 2}; do
  RES=(); [ -f "$ROOT/out/${TAG}_s$S.pkl" ] && RES=(-E RESUME=out/${TAG}_s$S.pkl)
  systemd-run --user --unit=organic-$TAG-s$S --collect -p WorkingDirectory="$ROOT" -p Nice=10 "${RES[@]}" "${SPEAK[@]}" $EXTRA \
    -E WORLD=2 -E PARROT=1 \
    -E OMP_NUM_THREADS=3 -E OPENBLAS_NUM_THREADS=3 -E SPEECH=1 -E WORDS=1 -E HUB=0 -E MAX_UNITS=2000 -E MAX_CUES=8192 -E OVERHEARD=1 -E OBSERVE=${OBSERVE:-1} -E JUDGE=${JUDGE:-1} \
    -E MOTOR_UNITS=300 -E MAX_OUTPUTS=1024 -E NEURONS=8 -E MAX_CLUSTERS=2000 -E MAX_GB=4 -E EVERY=5000 -E STAGE_BUDGET=${STAGE_BUDGET:-100000} -E TOTAL=${TOTAL:-400000} -E TEST_N=300 \
    -E CORTEX=data/visual_cortex.pkl -E CORTEX_GRID=4 -E CORTEX_LAYERS=0,1,2 -E SEED=$S \
    -p StandardOutput=append:"$ROOT/out/${TAG}_s$S.log" -p StandardError=append:"$ROOT/out/${TAG}_s$S.log" \
    "$ROOT/.venv/bin/python" experiments/curriculum_life.py out/${TAG}_s$S.jsonl
done
