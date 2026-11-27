"""Why did the ear's units not find the photographs? (docs/VISION.md 2026-11-27 14:24, said before it runs.)

experiments/grounded_speech_probe.py, seed 0: the photograph's own spoken caption among 1,000 at recall@10 0.014
(chance 0.010; median rank 452 against 502 with the captions shuffled). Three things could be the limit: the ear's
units (the cue), the photo cortex's code of a whole scene (the content), or the link between them (Hebb's rule from
bags that are not decorrelated). Here the same captions and photographs, the bags read back from the run, and a grid:
    cue    sound   the run's bags of two- and three-syllable kinds (nothing read)
           words   the same captions' written words as a bag: A MEASURE ONLY, the cue an ear with perfect words
                   would give; nothing of it is in any brain
    link   hebb            the run's: the running mean of the product, bags centred and at unit length
           hebb, relayed   the same after a relay evens the bags out too (Relay, as on the sight side: both codes
                           decorrelated, then Hebb)
           ridge           the reference: the best linear map from bag to photograph code (the regulariser chosen on
                           the test pairs: generous to the reference)
Test as the run's: 1,000 held-out photographs with one caption each, recall at 1, 5, 10 both ways, median rank.

    SEED=0 python experiments/grounded_diag_check.py
"""
import json, os, pickle, sys, time
import numpy as np
import scipy.linalg as sl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Relay  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
EAR_U, ASSOC_U, TEST, SEED, DEV = int(os.environ.get("EAR_U", 20000)), int(os.environ.get("ASSOC_U", 20000)), int(os.environ.get("TEST", 1000)), int(os.environ.get("SEED", 0)), int(os.environ.get("DEV", 16000))
READER, SUB = os.environ.get("READER", ""), int(os.environ.get("SUB", 0))  # the one-voice run's bags; only the first SUB captions for the link (15:19)
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


# the run's own choice of photographs and captions (the same calls on the same random state, in its order)
rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
if READER:
    count = {}
    for u in utts:
        count[u["speaker"]] = count.get(u["speaker"], 0) + 1
    who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1] if READER.startswith("top") else READER  # top, top2, top3: by captions read
    utts = [u for u in utts if u["speaker"] == who]
row_of = {i: r for r, i in enumerate(json.load(open(os.path.expanduser("~/datasets/coco_thumbs/train2017_ids.json"))))}
thumbs = np.load(os.path.expanduser("~/datasets/coco_thumbs/train2017_64.npy"), mmap_mode="r")
images = sorted({u["image"] for u in utts})
images = [images[i] for i in rng.permutation(len(images))]
test_imgs, assoc_imgs = images[:TEST], images[TEST: TEST + ASSOC_U]
by_image = {}
for u in utts:
    by_image.setdefault(u["image"], []).append(u)
test_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in test_imgs]
assoc_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in assoc_imgs]
del utts, by_image

z = np.load(f"out/bench/grounded_bags_s{SEED}_e{EAR_U}_a{ASSOC_U}_t{TEST}_d{DEV}{'_r' + READER if READER else ''}.npz")
if SUB:
    assoc_u, assoc_imgs = assoc_u[:SUB], assoc_imgs[:SUB]
cues = {"sound": (z["assoc"][:len(assoc_u)], z["test"])}
count = {}
for u in assoc_u:
    for w in set(u["text"].split()):
        count[w] = count.get(w, 0) + 1
vocab = {w: k for k, w in enumerate(sorted(w for w, n in count.items() if n >= 5))}


def words(us):
    B = np.zeros((len(us), len(vocab)), np.float32)
    for i, u in enumerate(us):
        for w in u["text"].split():
            if w in vocab:
                B[i, vocab[w]] += 1
    return B


cues["words"] = (words(assoc_u), words(test_u))
say(f"{len(assoc_u)} captions for the link, {len(test_u)} for the test; sound kinds {cues['sound'][0].shape[1]}, "
    f"written words heard in 5+ captions {len(vocab)}; kinds / words a caption: {cues['sound'][0].sum(1).mean():.1f} / {cues['words'][0].sum(1).mean():.1f}")

CODES = f"out/bench/grounded_codes_s{SEED}_a{ASSOC_U}_t{TEST}{'_r' + READER if READER else ''}.npz"
if os.path.exists(CODES):
    c = np.load(CODES)
    V_assoc, V_test = c["assoc"].astype(np.float32)[:len(assoc_u)], c["test"].astype(np.float32)
else:
    cortex = pickle.load(open(f"out/bench/cortex_ph_d_s{SEED}.pkl", "rb"))
    vrelay, cortex.relay = cortex.relay, None

    def see(ids):
        out = np.empty((len(ids), vrelay.Z.shape[1]), np.float32)
        for a in range(0, len(ids), 500):
            X = np.stack([cortex.whole(np.asarray(thumbs[row_of[i]], np.float32) / 255, 4) for i in ids[a: a + 500]])
            out[a: a + 500] = unit((X - vrelay.mu) @ vrelay.Z)
        return out
    V_assoc, V_test = see(assoc_imgs), see(test_imgs)
    np.savez(CODES, assoc=V_assoc.astype(np.float16), test=V_test.astype(np.float16))
    del cortex, vrelay
say("the photographs' codes")


def recall(P):
    """P: each test caption's predicted photograph code."""
    S = unit(P) @ V_test.T  # caption i against photograph j
    r_cap = (S > np.diag(S)[:, None]).sum(1)
    r_img = (S > np.diag(S)[None, :]).sum(0)
    rec = lambda r: " ".join(f"R@{k} {np.mean(r < k):.3f}" for k in (1, 5, 10))  # noqa: E731
    return f"photograph -> caption {rec(r_img)}; caption -> photograph {rec(r_cap)}; median rank {np.median(r_img) + 1:.0f} / {np.median(r_cap) + 1:.0f}", float(np.mean(r_img < 10))


PART = os.environ.get("PART", "all")  # "sight": only the last measure (the whole file at once passed a 2.8 GB cap)
for cue, (Ba, Bt) in (cues.items() if PART == "all" else ()):
    mu = Ba.mean(0)
    Bc, Btc = unit(Ba - mu), unit(Bt - mu)
    say(f"{cue:5} hebb         : " + recall(Btc @ (Bc.T @ V_assoc / len(Bc)))[0])
    own = Relay(0.0005).settle(Bc)
    Br, Btr = unit((Bc - own.mu) @ own.Z), unit((Btc - own.mu) @ own.Z)
    say(f"{cue:5} hebb, relayed: " + recall(Btr @ (Br.T @ V_assoc / len(Br)))[0])
    shuf = Br[np.random.default_rng(SEED + 1).permutation(len(Br))]
    say(f"{cue:5} hebb, relayed, captions shuffled among the photographs: " + recall(Btr @ (shuf.T @ V_assoc / len(Br)))[0])
    m = Bc.mean(0)
    A = ((Bc - m).T @ (Bc - m)).astype(np.float64)
    Y = (Bc - m).T @ (V_assoc - V_assoc.mean(0))
    best = ("", -1.0)
    for lam in (0.01, 0.1, 1.0, 10.0):
        W = sl.solve(A + lam * (np.trace(A) / len(A)) * np.eye(len(A)), Y, assume_a="pos")
        line, r10 = recall((Btc - m) @ W.astype(np.float32))
        if r10 > best[1]:
            best = (f"(lambda {lam}) " + line, r10)
    say(f"{cue:5} ridge        : " + best[0])

# (14:31) How much of the written words do the ear's bags carry at all, the eye left out? A measure only: ridge from a
# caption's sound bag to its bag of written words; each test caption's own words among the 1,000 by the prediction,
# and which words are heard best (the correlation of a word's predicted and true presence over the test captions).
(Sa, St), (Wa, Wt) = cues["sound"], cues["words"]
ms, mw = Sa.mean(0), Wa.mean(0)
Sc, Stc = unit(Sa - ms), unit(St - ms)
A = (Sc.T @ Sc).astype(np.float64)
Y = Sc.T @ (Wa - mw)
V_test = unit(Wt - mw)  # recall() now scores against the test captions' own word bags
inv = {k: w for w, k in vocab.items()}
often = np.nonzero((Wt > 0).sum(0) >= 30)[0]
for lam in ((0.1, 10.0, 1.0) if PART == "all" else (1.0,)):  # (1.0 last: its reading of the words is used below)
    P = Stc @ sl.solve(A + lam * (np.trace(A) / len(A)) * np.eye(len(A)), Y, assume_a="pos").astype(np.float32)
    r = np.array([np.corrcoef(P[:, k], Wt[:, k] > 0)[0, 1] for k in often])
    top = np.argsort(-r)[:20]
    say(f"sound bag -> the caption's written words, ridge (lambda {lam}): its own words among {TEST}: {recall(P)[0].replace('photograph', 'words').replace('caption', 'sound')} | "
        f"{len(often)} words in 30+ test captions, correlation predicted/true: median {np.median(r):.2f}, above 0.3: {int((r > 0.3).sum())}; best: "
        + ", ".join(f"{inv[often[k]].lower()} {r[k]:.2f}" for k in top))

del P, Sc, Sa
# (14:34) Are the words the ear hears enough to find a photograph? Measures only (text in every step).
c = np.load(CODES)
V_assoc, V_test = c["assoc"].astype(np.float32)[:len(assoc_u)], c["test"].astype(np.float32)
mv = V_assoc.mean(0)


def to_sight(Ba, lam=10.0):
    m = Ba.mean(0)
    Bc = unit(Ba - m)
    mm = Bc.mean(0)
    G = ((Bc - mm).T @ (Bc - mm)).astype(np.float64)
    W = sl.solve(G + lam * (np.trace(G) / len(G)) * np.eye(len(G)), (Bc - mm).T @ (V_assoc - mv), assume_a="pos").astype(np.float32)
    return lambda B: (unit(B - m) - mm) @ W


f = to_sight(Wa)
say("written words -> photograph (ridge, as above)          : " + recall(f(Wt))[0])
heard = (Stc @ sl.solve(A + 1.0 * (np.trace(A) / len(A)) * np.eye(len(A)), Y, assume_a="pos").astype(np.float32)) + mw
say("sound -> predicted words -> photograph (the same map)  : " + recall(f(heard))[0])
for name, keep in ((f"the {len(often)} words in 30+ test captions", often), ("the 20 heard best", often[np.argsort(-r)[:20]]),
                   ("all words but those " + str(len(often)), np.setdiff1d(np.arange(len(vocab)), often))):
    say(f"written words, only {name}: " + recall(to_sight(Wa[:, keep])(Wt[:, keep]))[0])

# (14:35) How well is a word heard, by how often it is said? (the reading of lambda 1.0, the correlation as above)
n_test = (Wt > 0).sum(0)
for lo, hi in ((30, 10**6), (15, 30), (8, 15), (4, 8)):
    ks = np.nonzero((n_test >= lo) & (n_test < hi))[0]
    rr = np.array([np.corrcoef(heard[:, k], Wt[:, k] > 0)[0, 1] for k in ks])
    best = np.argsort(-rr)[:8]
    say(f"words in {lo}-{min(hi, int(n_test.max())) if hi > 10**5 else hi - 1} of the {TEST} test captions: {len(ks)} words, heard at correlation median {np.median(rr):.2f}, above 0.3: {int((rr > 0.3).sum())}; best: "
        + ", ".join(f"{inv[ks[k]].lower()} {rr[k]:.2f}" for k in best))
