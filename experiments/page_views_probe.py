"""A thing on a page, view by view (docs/VISION.md, 2026-11-23; an observation of saved brains, nothing learns).

The ablation lives: with a never-developed cortex a thing on a page is named 0.04 above the floor (developed: 0.36),
the level of a brain that does not look, although the gaze does not use the cortex's kinds and a probe that isolates
the fixated thing names it nearly as well either way. So what does the brain make of each view of a test page?
For a saved brain and the held-out "what is this?" pages of the pages stage:
    as it lives     the brain's own answer, looking as it does (the whole page seen coarsely, and its fixations)
    store silent    the same, with the episodic store taken out for the asking
    whole only      the answer from the coarse whole page alone, not looking
    each fixation   the answer from one fixated window alone, filling the view (as the thing looks alone)
and, counted over pages: how many fixations there were; whether any fixation alone names the thing; whether the
whole page alone does; and when the brain is wrong while a fixation alone is right, what it said instead.
Also, for scale: single pictures of first words, named not looking.
    python experiments/page_views_probe.py out/cleanw_s0.pkl out/ablrand_s0.pkl
"""
import collections, os, pickle, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402
from organic.gaze import fixations  # noqa: E402

cur = Curriculum(seed=0, world=4)
N = int(os.environ.get("N", 300))
pages = [m for m in cur.test_set("W2", n=N) if m["source"] == "page:name"]
singles = [m for m in cur.test_set("W1", n=N) if m["source"].endswith(":name")]
for path in sys.argv[1:]:
    brain = pickle.load(open(path, "rb"))
    brain.learning = False
    say = lambda img, q: tidy(brain.live(np.asarray(img, np.float32), q, learn=False, explore=False)[0] or "")  # noqa: E731
    right = lambda said, m: grade("W2", said, m["answers"]) >= 0.5  # noqa: E731
    looking = brain.looking
    rows, said_instead, blends = [], collections.Counter(), 0
    for m in pages:
        scene = np.asarray(m["image"], np.float32)
        side = scene.shape[0] // 2
        fill = lambda w: w if side == brain.view else np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize((brain.view, brain.view), Image.BILINEAR), np.float32) / 255  # noqa: E731
        fix = list(fixations(scene, most=4, apart=brain.apart))
        brain.looking = looking
        full = say(scene, m["question"])
        brain.looking = False
        whole = say(brain._coarse(scene), m["question"])
        each = [say(fill(scene[r : r + side, c : c + side]), m["question"]) for r, c in fix]
        rows.append((right(full, m), right(whole, m), [right(e, m) for e in each], len(fix)))
        names = set(each) | {whole}
        if not right(full, m) and any(right(e, m) for e in each):
            said_instead["the whole page's answer" if full == whole else "another fixation's answer" if full in each else "a run of several views' answers" if any(n and n != full and n in full for n in names) else "something else"] += 1
        blends += int(any(n and n != full and n in full for n in names))
    brain.looking, store = looking, brain.words.episodes
    brain.words.episodes = None  # the same brain, its episodic store silent
    silent = np.mean([right(say(np.asarray(m["image"], np.float32), m["question"]), m) for m in pages])
    brain.words.episodes = store
    brain.looking = False
    alone = np.mean([grade("W1", say(m["image"], m["question"]), m["answers"]) >= 0.5 for m in singles])
    brain.looking = looking
    full, whole = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
    anyfix = np.mean([any(r[2]) for r in rows])
    perfix = np.mean([x for r in rows for x in r[2]]) if any(r[3] for r in rows) else float("nan")
    print(f"{path} (age {brain.age}; looking {looking}; {len(pages)} pages, {len(singles)} single pictures)\n"
          f"   single pictures, not looking {alone:.2f} | a thing on a page: as it lives {full:.2f} | as it lives, the store silent {silent:.2f} | the whole page alone {whole:.2f} | some fixation alone {anyfix:.2f} | a fixation alone, on average {perfix:.2f}\n"
          f"   fixations a page: {dict(sorted(collections.Counter(r[3] for r in rows).items()))} | answers that run several views' answers together {blends / len(rows):.2f}\n"
          f"   wrong as it lives although a fixation alone is right: {sum(said_instead.values())} pages; it said: {dict(said_instead)}", flush=True)
    del brain
