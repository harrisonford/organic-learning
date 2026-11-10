"""Observation before interrogation: do names form from hearing adults talk about pictures, unasked? (offline)

docs/REVIEW_2026-11-10.md 4.4. Children mostly learn words by observation: speech about what they are
looking at, with no question and no answer expected. For the word cells (organic/words.py) an observation is
nothing special: an utterance heard after silence while a picture is seen, so each unit cell's dendrites
learn to predict it from the picture, as they do a reply's.

The first-words pictures are split: a part is ASKED about (question -> the adult's answer, as every
moment so far), the rest are only OBSERVED ("look, a dog!", data/world/observation_frames.json: 66 ways,
written by the adults). Three sets of word cells:
  asked only          lessons on the asked part; the rest of the pictures are never seen
  asked + observed    the same lessons, and the adult's unasked comments on the other pictures
  all asked           lessons on every picture (what full interrogation gives)
Then held-out pictures are asked about, three wordings each. An answer is right if its content words are
the adult's ("dog!", "a dog" = "dog"), as the judge accepts them.
The ear first hears the comments for a while (as an infant would, before words), then is kept still.
    ASKED=0.1 python experiments/observation_probe.py
"""
import copy, json, os, pickle, random, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402
from organic.judge import _core  # noqa: E402
from organic.world import WORLD_DIR  # noqa: E402

ASKED = float(os.environ.get("ASKED", 0.1))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
cur = Curriculum(seed=0)
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
FRAMES = json.load(open(os.path.join(WORLD_DIR, "naming_frames.json")))["frames"]
V = {}


def vis(ref):
    if ref not in V:
        v = np.concatenate(cortex.code(cur.image(ref), grid=4))
        V[ref] = v / (np.linalg.norm(v) + 1e-8)
    return V[ref]


def comment(w, r):
    return r.choice(cur.comments).replace("a THING", ("an " if w[0] in "aeiou" else "a ") + w).replace("THING", w)


pics = [it for src in ("cifar:name", "commons:name") for it in cur.train["W1"][src]]
adult = cur.train["W1"].get("curriculum", [])
test_pics = [it for src in ("cifar:name", "commons:name") for it in cur.test["W1"][src][:150]]
ear = pickle.load(open(os.environ.get("EAR", "out/kid_s3.ear.pkl"), "rb"))
r0 = random.Random(99)
for _ in range(int(os.environ.get("EAR_HEARS", 20000))):  # the ear hears adults comment on pictures (and learns), before the probe
    ear.units(comment(r0.choice(pics)[2][0], r0))
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
print("the ear now cuts:", [[u.decode() for u in hear(s)] for s in ("look, a dog!", "that's a cat", "here is the elephant", "a turtle")], flush=True)
results = {}
for seed in SEEDS:
    r = random.Random(seed)
    order = pics[:]
    r.shuffle(order)
    n_ask = int(ASKED * len(order))
    asked, observed = order[:n_ask], order[n_ask:]
    arms = {"asked only": [("ask", it) for it in asked + adult],
            "asked + observed": [("ask", it) for it in asked + adult] + [("see", it) for it in observed],
            "all asked": [("ask", it) for it in order + adult]}
    for name, moments in arms.items():
        r2 = random.Random(seed + 1)
        r2.shuffle(moments)
        t = Words(len(vis(pics[0][0])), ear)
        t0 = time.time()
        for kind, (ref, q, ans, st, src) in moments:
            v = vis(ref) if ref[0] != "dark" else None
            if kind == "ask":
                t.learn(hear(cur.wording((ref, q, ans, st, src), r2)), v, hear(ans[0]))
            else:  # an utterance heard after silence, while looking
                t.learn([], v, hear(comment(ans[0], r2)))
        right, same, every, said_all = [], 0, 0, []
        for it in test_pics:
            asks = [FRAMES[(zlib.crc32(repr(it[0]).encode()) + 17 * j + 5) % len(FRAMES)] for j in range(3)]
            said = [t.recall(hear(q), vis(it[0]))[0] or "" for q in asks]
            ok = [bool(_core(s)) and _core(s) in [_core(a) for a in it[2]] for s in said]
            right += ok
            same += len(set(said)) == 1
            every += all(ok)
            said_all += said
        row = [float(np.mean(right)), same / len(test_pics), every / len(test_pics)]
        results.setdefault(name, []).append(row)
        print(f"seed {seed} {name:17} {len(moments):>6} moments, {t.cells()}, {time.time() - t0:.0f}s | "
              f"right {row[0]:.3f} | same x3 {row[1]:.3f} | right x3 {row[2]:.3f} | says e.g. {said_all[:8]}", flush=True)
print(f"\nASKED={ASKED}: mean over seeds {SEEDS} (spread between seeds)")
for name, rows in results.items():
    m, s = np.mean(rows, axis=0), np.std(rows, axis=0)
    print(f"  {name:17} right {m[0]:.3f}±{s[0]:.3f} | same x3 {m[1]:.3f}±{s[1]:.3f} | right x3 {m[2]:.3f}±{s[2]:.3f}")
json.dump(results, open("out/observation_probe.json", "w"))
