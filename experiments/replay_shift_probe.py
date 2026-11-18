"""Does replay from the fast store keep what a stage change erases? (docs/RESEARCH_MEMORY.md, 2026-11-18)

A life loses what it is no longer taught: eyes2_s1 named a thing on a page 0.41 when it passed two words and 0.2
after 30,000 moments of preschool; the first world 3 lives lost their COCO names the same way (erasure in the word
cells' synapses, forget_diag.py). Complementary learning systems says what the remedy is: the fast store replays
old moments to the slow learner while new ones are lived.

Here the word cells alone (new cortex and relay, full-size corrections, as the lives), on cached codes:
  stage A   NA naming moments, the three sources taking turns (CIFAR, Commons, COCO things)
  stage B   NB naming moments of CIFAR and Commons only: COCO is no longer taught
and held-out naming by source after A and after B. Arms:
  none       the word cells alone
  store      with the fast store as the lives have it (the oldest moment goes; no replay)
  fifoR      that store, and every REST moments R stored moments come back at random (replay of the recent only)
  replayR    the store with replay as built on 2026-11-18 (organic/episodes.py): strong moments come back more often,
             a moment is held as far as the cells were surprised by it when it came back, the weakest goes
  rehearseR  the reference: R moments of stage A's own lessons lived again every REST moments (a perfect memory)
  joint      the upper reference: COCO goes on being taught in stage B
    SEEDS=0 ARMS=none,store,fifo30,replay10,replay30,rehearse30,joint python experiments/replay_shift_probe.py
"""
import os, pickle, random, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.episodes import Episodes  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ARMS = os.environ.get("ARMS", "none,store,fifo30,replay10,replay30,rehearse30,joint").split(",")
NA, NB, REST = int(os.environ.get("NA", 6000)), int(os.environ.get("NB", 6000)), int(os.environ.get("REST", 100))
CAP, FADE = int(os.environ.get("CAP", 2000)), float(os.environ.get("FADE", 2000))
PATH = os.environ.get("CORTEX", "data/visual_cortex2.pkl")
cur = Curriculum(seed=0, world=3)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
SRC = ("cifar:name", "commons:name", "coco:name")
lived = {s: [it for it in cur.train["W1"][s] if cur.image(it[0]).max() > 0] for s in SRC}
held = {s: [it for it in cur.test["W1"][s] if cur.image(it[0]).max() > 0][:200] for s in SRC}
cortex = pickle.load(open(PATH, "rb"))
cpath = f"out/tmp/woc_{os.path.basename(PATH)}"
cache = pickle.load(open(cpath, "rb")) if os.path.exists(cpath) else {}
had = len(cache)


def code(ref):
    if ref not in cache:
        cache[ref] = cortex.whole(cur.image(ref), 4).astype(np.float32)
    return cache[ref].astype(np.float32)


def named(words):
    ask = hear("what is this?")
    return {s: float(np.mean([tidy(words.recall(ask, code(it[0]))[0] or "") == it[2][0] for it in held[s]])) for s in SRC}


for seed in SEEDS:
    for arm in ARMS:
        rng, t0 = random.Random(seed), time.time()
        A = [(rng.choice(lived[rng.choice(SRC)]), rng.choice(cur.frames)) for _ in range(NA)]
        B = [(rng.choice(lived[rng.choice(SRC if arm == "joint" else SRC[:2])]), rng.choice(cur.frames)) for _ in range(NB)]
        dose = int(re.sub(r"\D", "", arm) or 0)
        words = Words(cortex.relay.Z.shape[0] if getattr(cortex, "relay", None) is not None else len(code(A[0][0][0])), ear)
        words.plastic_by_error = False
        if arm != "none" and not arm.startswith("rehearse") and arm != "joint":
            words.episodes = Episodes(words.visual_dim, ear.width, capacity=CAP)
            if arm.startswith("replay"):
                words.episodes.fade = FADE
        ep, np_rng, rr = words.episodes, np.random.default_rng(seed), random.Random(seed + 7)
        tags = np.zeros(CAP, np.int8)  # which stage each stored moment is from (1: A's COCO, 2: A's others, 3: B)
        came = [0, 0, 0, 0]

        def live(it, frame, tag):
            words.learn(hear(frame), code(it[0]), hear(it[2][0]))
            if ep is not None and ep.last is not None:
                tags[ep.last] = tag

        def rest():
            if arm.startswith("rehearse"):
                for it, frame in rr.sample(A, dose):
                    words.learn(hear(frame), code(it[0]), hear(it[2][0]))
            elif dose and ep is not None:
                if arm.startswith("fifo"):  # at random among what is stored, nothing kept for having come back
                    for k in rr.sample(range(ep.n), min(dose, ep.n)):
                        came[tags[k]] += 1
                        words.learn(list(ep.asked[k]), ep.seen[k].astype(np.float32), list(ep.reply[k]), again=k)
                        ep.strength[k] = max(ep.strength[k], 1e-3)  # (lived_again set it; the fifo store does not use it to keep)
                else:
                    before = ep.clock
                    back = ep.come_back(np_rng, dose)
                    for k in back:
                        came[tags[k]] += 1
                        words.look(None)
                        words.learn(list(ep.asked[k]), ep.seen[k].astype(np.float32), list(ep.reply[k]), again=k)
                    assert ep.clock == before  # a moment that came back is not written again

        for n, (it, frame) in enumerate(A):
            live(it, frame, 1 if it[4] == "coco:name" else 2)
            if n % REST == REST - 1:
                rest()
        after_a = named(words)
        for n, (it, frame) in enumerate(B):
            live(it, frame, 3)
            if n % REST == REST - 1:
                rest()
        after_b = named(words)
        kept = "" if ep is None else f" | the store at the end: A's COCO {int((tags[: ep.n] == 1).sum())}, A's others {int((tags[: ep.n] == 2).sum())}, B {int((tags[: ep.n] == 3).sum())}; came back: A's COCO {came[1]}, A's others {came[2]}, B {came[3]}"
        print(f"seed {seed} {arm:11} ({time.time() - t0:.0f}s): after A " + " ".join(f"{s.split(':')[0]} {after_a[s]:.3f}" for s in SRC)
              + " | after B " + " ".join(f"{s.split(':')[0]} {after_b[s]:.3f}" for s in SRC)
              + f" | COCO kept {after_b['coco:name'] - after_a['coco:name']:+.3f}" + kept, flush=True)
if len(cache) > had:
    pickle.dump(cache, open(cpath, "wb"))
