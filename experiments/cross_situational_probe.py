"""Cross-situational links vs whole-situation memory, on first words (offline).

Trouble 3 (docs/PLASTICITY.md, the chat of 2026-11-06): a cluster stores a whole situation
(the question's sound + the picture -> the answer), so asking about the same picture in
other words misses it. Children learn word-thing links across situations (Smith & Yu
2008): the word and the thing keep co-occurring, while "what", "look", "is" co-occur with
everything and so favour nothing. Nobody marks which words matter.

  whole situation   nearest remembered moment on [question code ; visual code] (roughly the clusters)
  vision only       nearest remembered picture (the eye's ceiling for naming)
  cross, visual cells  the same links, but vision first forms its own cells (vision-only memories) and
                    those cells link to words across situations
  cross-situational every active input (each heard sound unit, each visual feature) keeps Hebbian
                    links to the words it co-occurs with, normalised by how often it is active
                    (its links become P(word | input)); to answer, active inputs vote with their links,
                    each organ casting one vote in total. An input that occurs with every word has flat
                    links and favours nothing: frame-ignoring has to emerge, it is not written anywhere.

Measured on W1 test pictures (CIFAR, Commons), each asked three different ways (naming frames):
accuracy, the same answer all three times, right all three times; and on W1 adult items
(the informative word is in the question: "what does the cow say?").
    python experiments/cross_situational_probe.py
"""
import collections, json, os, pickle, sys, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402
from organic.world import WORLD_DIR  # noqa: E402

cur = Curriculum(seed=0)
ear = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/kid_s3.ear.pkl", "rb"))
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
FRAMES = json.load(open(os.path.join(WORLD_DIR, "naming_frames.json")))["frames"]
PICS = ("cifar:name", "commons:name")


def vis(ref):
    img = cur.image(ref)
    if img.max() == 0:
        return None
    v = np.concatenate(cortex.code(img, grid=4))
    return v / (np.linalg.norm(v) + 1e-8)


def heard(q):
    return ear.units(q, learn=False)


train = [it for src, items in cur.train["W1"].items() for it in items]
test_pics = [it for src in PICS for it in cur.test["W1"][src][:150]]
test_text = cur.test["W1"].get("curriculum", [])[:150]
print(f"train {len(train)} moments, test {len(test_pics)} pictures x 3 wordings, {len(test_text)} adult items", flush=True)

words = sorted({it[2][0] for it in train})
W = {w: i for i, w in enumerate(words)}
V = {}  # visual code cache
for it in train + test_pics:
    if it[0] not in V:
        V[it[0]] = vis(it[0])
dim_v = next(v for v in V.values() if v is not None).size

# ---- cross-situational links
S = collections.defaultdict(lambda: np.zeros(len(words)))  # sound unit -> co-occurrence with words
Sv, Nv = np.zeros((dim_v, len(words))), np.zeros(dim_v)
# ---- whole-situation memory
mem_q, mem_v, mem_w = [], [], []
for ref, q, ans, st, src in train:
    w = W[ans[0]]
    for u in set(heard(q)):
        S[u][w] += 1
    v = V[ref]
    if v is not None:
        Sv[:, w] += v
        Nv += v
    mem_q.append(ear.question(heard(q), adapt=False))
    mem_v.append(v if v is not None else np.zeros(dim_v))
    mem_w.append(ans[0])
mem_q, mem_v = np.array(mem_q), np.array(mem_v)
has_v = np.linalg.norm(mem_v, axis=1) > 0
P_s = {u: c / c.sum() for u, c in S.items()}
P_v = Sv / (Nv[:, None] + 1e-8)


def cross(q, v):
    votes = np.zeros(len(words))
    units = [u for u in set(heard(q)) if u in P_s]
    if units:
        votes += np.mean([P_s[u] for u in units], axis=0)
    if v is not None:
        votes += (v @ P_v) / (v.sum() + 1e-8)
    return words[int(np.argmax(votes))]


def cross_cells(q, v, k=5):
    """The same links, but vision first forms its own cells (one per remembered picture, vision only, as clusters
    grow), and the k cells that answer a picture link to words across situations (their word co-occurrences)."""
    votes = np.zeros(len(words))
    units = [u for u in set(heard(q)) if u in P_s]
    if units:
        votes += np.mean([P_s[u] for u in units], axis=0)
    if v is not None:
        sim = np.where(has_v, mem_v @ v, -9)
        for j in np.argpartition(sim, -k)[-k:]:
            votes[W[mem_w[j]]] += 1.0 / k
    return words[int(np.argmax(votes))]


def whole(q, v):
    qc = ear.question(heard(q), adapt=False)
    sim = mem_q @ qc + (mem_v @ v if v is not None else 0)
    return mem_w[int(np.argmax(sim))]


def vision_only(q, v):
    if v is None:
        return ""
    sim = np.where(has_v, mem_v @ v, -9)
    return mem_w[int(np.argmax(sim))]


def wordings(it):
    return [FRAMES[(zlib.crc32(repr(it[0]).encode()) + 17 * j + 5) % len(FRAMES)] for j in range(3)]


for name, f in (("whole situation", whole), ("vision only", vision_only), ("cross-situational", cross), ("cross, visual cells", cross_cells)):
    acc, same, every = [], 0, 0
    for it in test_pics:
        said = [f(q, V[it[0]]) for q in wordings(it)]
        ok = [tidy(s) in [tidy(a) for a in it[2]] for s in said]
        acc += ok
        same += len(set(said)) == 1
        every += all(ok)
    txt = np.mean([grade("W1", f(it[1], None), it[2]) for it in test_text]) if test_text else float("nan")
    print(f"{name:18} pictures: right {np.mean(acc):.3f} | same answer x3 {same / len(test_pics):.3f} | right x3 {every / len(test_pics):.3f}"
          f" || adult items (word in the question): {txt:.3f}", flush=True)

# what the links learned: how much each heard unit favours one word (1 = always the same word, ~0 = flat)
spread = {u.decode(errors="replace"): float(p.max() - np.sort(p)[-2]) for u, p in P_s.items() if S[u].sum() >= 20}
flat = sorted(spread, key=spread.get)[:12]
sharp = sorted(spread, key=spread.get)[-12:]
print("flattest heard units (favour no word):", flat)
print("sharpest heard units (favour one word):", sharp)
