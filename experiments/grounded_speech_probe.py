"""Do the ear's own units go with what the pictures show? Spoken captions and photographs, no text (queue E 32;
docs/VISION.md 2026-11-26 23:10).

The ear of experiments/sound_units_probe.py (second reading: adapted to the voice, 8 band stretches, syllables by the
loudness rhythm, syllable kinds and two- and three-syllable kinds by the eye's rule) found kinds of sound that come
with one written word. Here the same ear, developed on EAR_U spoken COCO captions, hears ASSOC_U more, each with its
photograph (our 64 x 64 COCO thumbnails; SpokenCOCO's training images are all among them); a caption is the bag of
two- and three-syllable kinds heard in it. The link between the photograph's code (the photo cortex with its relay)
and the bag is Hebb's rule (the running mean of the product of the two, each centred and at unit length). Test, the
paradigm's own (visually grounded speech: Harwath, Torralba & Glass 2016; Harwath et al. 2018): 1,000 held-out
photographs and one spoken caption each, never heard; for each photograph, the rank of its own caption among the
1,000 by the link, and the other way; recall at 1, 5 and 10 (chance 0.001, 0.005, 0.01); beside it the same with the
bags shuffled among the photographs before the link is learned. The transcripts are not read at all.

    EAR_U=20000 ASSOC_U=20000 SEED=0 python experiments/grounded_speech_probe.py
    READER=top EAR_U=4670 ASSOC_U=5000 ...   the same in one voice: the reader with most captions (2026-11-27 15:19)
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2, Relay  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
EAR_U, ASSOC_U, TEST, SEED = int(os.environ.get("EAR_U", 20000)), int(os.environ.get("ASSOC_U", 20000)), int(os.environ.get("TEST", 1000)), int(os.environ.get("SEED", 0))
DEV, KINDS, WINDOW, BANDS, HALF, SMOOTH, K1, K23 = int(os.environ.get("DEV", 16000)), 128, 8, 8, 5, 5, 512, 1024
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
all_utts = utts  # (kept for OTHER, below)
READER = os.environ.get("READER", "")  # "top": the captions of the reader who read most, only (an ear that grows up in one voice; VISION.md 2026-11-27 15:19)
if READER:
    count = {}
    for u in utts:
        count[u["speaker"]] = count.get(u["speaker"], 0) + 1
    who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1] if READER.startswith("top") else READER  # top, top2, top3: by captions read
    utts = [u for u in utts if u["speaker"] == who]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
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
ear_pool = [u for i in images[TEST + ASSOC_U:] for u in by_image[i]]
ear_u = [ear_pool[i] for i in rng.choice(len(ear_pool), min(EAR_U, len(ear_pool)), replace=False)]
say(f"{len(utts)} spoken captions of {len(images)} photographs; ear develops on {len(ear_u)}, the link on {len(assoc_u)}, test {len(test_u)} (photographs never seen or heard)")


def raw(u):
    return np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)


def gram(u):
    g = raw(u)
    return (g - g.mean(0)) / (g.std(0) + 1e-3)  # adapted to the voice


layer = Layer2(1, KINDS, WINDOW, 1, seed=SEED)


def clip(u):
    g = gram(u)
    s = int(rng.integers(0, max(1, len(g) - 100)))
    return g[s: s + 100][..., None]


# (2026-11-27) the captions' bags are kept on disk once made: the first run was killed for memory after seven hours,
# coding the photographs, and lost them. A run that finds them (same sizes and seed) reads them back, with the random
# state as it was, and goes straight to the photographs.
BAGS = f"out/bench/grounded_bags_s{SEED}_e{EAR_U}_a{ASSOC_U}_t{TEST}_d{DEV}{'_r' + READER if READER else ''}.npz"
RESUME = os.path.exists(BAGS) and not os.environ.get("FRESH")  # FRESH=1: hear again (to keep the ear, which runs before 16:45 did not)
if not RESUME:
    layer.settle((clip(ear_u[i]) for i in rng.choice(len(ear_u), min(800, len(ear_u)), replace=False)), rng)
    for n in range(DEV):
        layer.learn(clip(ear_u[rng.integers(len(ear_u))]), rng, rate=0.05 * (1 - n / DEV) + 0.005)


def frames(u):
    r = layer.respond(gram(u)[..., None], pooled=False)
    b = r.shape[1] // BANDS
    codes = r[:, : BANDS * b].reshape(r.shape[0], BANDS, b, KINDS).mean(axis=2).reshape(r.shape[0], -1)
    return codes, np.convolve(raw(u).sum(axis=1), np.ones(WINDOW) / WINDOW, "valid")[: len(codes)]


if not RESUME:
    relay = Relay(0.0005).settle(np.concatenate([unit(frames(ear_u[i])[0])[::5] for i in rng.choice(len(ear_u), min(400, len(ear_u)), replace=False)]))
    say("hearing cortex and its relay")


def stretches(u):
    c, loud = frames(u)
    C = unit((unit(c) - relay.mu) @ relay.Z)
    env = np.convolve(loud, np.ones(SMOOTH) / SMOOTH, "same")
    quiet = env < 0.1 * env.max()
    troughs = [t for t in range(1, len(env) - 1) if env[t] == env[max(0, t - HALF): t + HALF + 1].min()]
    cuts = sorted(set(troughs) | {t for t in range(1, len(env)) if quiet[t] != quiet[t - 1]} | {0, len(env)})
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 5 or quiet[a:b].mean() > 0.5:
            if out:
                out[-1][1] = True
            continue
        h = (a + b) // 2
        out.append([np.concatenate([unit(C[a:h].mean(0)), unit(C[h:b].mean(0))]) / np.sqrt(2), False])
    return out


class Kinds:
    def __init__(self, n, dim):
        self.U, self.wins, self.founded = np.zeros((n, dim), np.float32), np.zeros(n), 0

    def __call__(self, s, learning):
        if learning and self.founded < len(self.U):
            self.U[self.founded], self.wins[self.founded] = s, 1
            self.founded += 1
            return self.founded - 1
        f = self.founded
        c = int(np.argmax(self.U[:f] @ s - (0.3 * (self.wins[:f] / (self.wins[:f].mean() + 1e-8) - 1) if learning else 0)))
        if learning:
            self.wins[c] += 1
            w = self.U[c] + (1.0 / (1 + self.wins[c] / 5)) * (s - self.U[c])
            self.U[c] = w / (np.linalg.norm(w) + 1e-8)
        return c

    def many(self, S):
        """Nothing learning: the winners of many stretches at once (the same as one at a time; 2026-11-27)."""
        return np.argmax(self.U[: self.founded] @ S.T, axis=0)


D = 2 * KINDS * BANDS
pairs, triples = Kinds(K23, 2 * D), Kinds(K23, 3 * D)


def bag(u, learning):
    ss = stretches(u)
    v, gap = [s for s, _ in ss], [g for _, g in ss]
    out = np.zeros(2 * K23, np.float32)
    if not learning and not os.environ.get("ONE_BY_ONE"):
        P = [np.concatenate(v[i: i + 2]) / np.sqrt(2) for i in range(len(v) - 1) if not gap[i]]
        T = [np.concatenate(v[i: i + 3]) / np.sqrt(3) for i in range(len(v) - 2) if not (gap[i] or gap[i + 1])]
        if P:
            np.add.at(out, pairs.many(np.stack(P)), 1)
        if T:
            np.add.at(out, K23 + triples.many(np.stack(T)), 1)
        return out
    for i in range(len(v) - 1):
        if not gap[i]:
            out[pairs(np.concatenate(v[i: i + 2]) / np.sqrt(2), learning)] += 1
    for i in range(len(v) - 2):
        if not (gap[i] or gap[i + 1]):
            out[K23 + triples(np.concatenate(v[i: i + 3]) / np.sqrt(3), learning)] += 1
    return out


if RESUME:
    z = np.load(BAGS)
    B_assoc, B_test = z["assoc"], z["test"]
    rng.bit_generator.state = json.loads(str(z["rng"]))
    say(f"the captions' bags of sound kinds, read back from {BAGS}")
else:
    for i, u in enumerate(ear_u):
        bag(u, True)
        if (i + 1) % 5000 == 0:
            say(f"ear: {i + 1} captions heard")
    B_assoc = np.zeros((len(assoc_u), 2 * K23), np.float32)
    for i, u in enumerate(assoc_u):
        B_assoc[i] = bag(u, False)
        if (i + 1) % 5000 == 0:
            say(f"bags: {i + 1} captions")
    B_test = np.stack([bag(u, False) for u in test_u])
    np.savez(BAGS, assoc=B_assoc, test=B_test, rng=json.dumps(rng.bit_generator.state))
    say(f"the captions' bags of sound kinds (kept in {BAGS})")
    EAR = BAGS.replace("grounded_bags", "grounded_ear").replace(".npz", ".pkl")  # (2026-11-27 16:45) the ear kept, for OTHER
    pickle.dump({"layer": layer, "relay": relay, "pairs": (pairs.U, pairs.wins, pairs.founded), "triples": (triples.U, triples.wins, triples.founded)}, open(EAR, "wb"))
    if not os.environ.get("OTHER"):
        del layer, relay, pairs, triples

cortex = pickle.load(open(f"out/bench/cortex_ph_d_s{SEED}.pkl", "rb"))
vrelay, cortex.relay = cortex.relay, None


def see(ids):  # a few hundred photographs at a time (the whole stack at once took the first run past its memory)
    out = np.empty((len(ids), vrelay.Z.shape[1]), np.float32)
    for a in range(0, len(ids), 500):
        X = np.stack([cortex.whole(np.asarray(thumbs[row_of[i]], np.float32) / 255, 4) for i in ids[a: a + 500]])
        out[a: a + 500] = unit((X - vrelay.mu) @ vrelay.Z)
    return out


V_assoc, V_test = see(assoc_imgs), see(test_imgs)
say("the photographs' codes")

mu_b = B_assoc.mean(0)
Bc, Bt = unit(B_assoc - mu_b), unit(B_test - mu_b)
for name, Bl in (("own captions", Bc), ("captions shuffled among the photographs", Bc[rng.permutation(len(Bc))])):
    M = Bl.T @ V_assoc / len(Bl)  # Hebb: the running mean of the product
    S = unit(Bt @ M) @ V_test.T  # caption i against photograph j
    r_cap = (S > np.diag(S)[:, None]).sum(1)  # for each caption: photographs scored above its own
    r_img = (S > np.diag(S)[None, :]).sum(0)  # for each photograph: captions scored above its own
    rec = lambda r: " ".join(f"R@{k} {np.mean(r < k):.3f}" for k in (1, 5, 10))  # noqa: E731
    say(f"{name}: photograph -> its caption among {TEST}: {rec(r_img)}; caption -> its photograph: {rec(r_cap)}; median rank {np.median(r_img) + 1:.0f} / {np.median(r_cap) + 1:.0f} (chance R@10 {10 / TEST:.3f}, median {TEST // 2})")

# OTHER=top2 (VISION.md 2026-11-27 16:45, said before it runs): how far does a one-voice ear carry to a voice it never
# heard? The ear and the link as they are (nothing learns), and 1,000 captions of another reader about photographs
# the link never saw; the same test as above. Beside it, how well each voice's bags give the caption's written words
# (a measure only: ridge from this reader's link captions' bags to their words, read on both test sets).
OTHER = os.environ.get("OTHER", "")
if OTHER:
    if RESUME:
        ear = pickle.load(open(BAGS.replace("grounded_bags", "grounded_ear").replace(".npz", ".pkl"), "rb"))
        layer, relay = ear["layer"], ear["relay"]
        (pairs.U, pairs.wins, pairs.founded), (triples.U, triples.wins, triples.founded) = ear["pairs"], ear["triples"]
    count = {}
    for u in all_utts:
        count[u["speaker"]] = count.get(u["speaker"], 0) + 1
    who2 = sorted(count, key=count.get, reverse=True)[int(OTHER[3:] or 1) - 1] if OTHER.startswith("top") else OTHER
    known, picked, other_u = set(assoc_imgs) | set(test_imgs), set(), []
    for u in all_utts:
        if u["speaker"] == who2 and u["image"] not in known and u["image"] not in picked:
            other_u.append(u)
            picked.add(u["image"])
    order = rng.permutation(len(other_u))
    LINK2 = int(os.environ.get("OTHER_LINK", 0))  # (17:40) this ear, its link learned again in the other voice from LINK2 pairs
    link_u = [other_u[i] for i in order[TEST: TEST + LINK2]]
    other_u = [other_u[i] for i in order[:TEST]]
    if os.environ.get("OTHER_RELAY") == "own":  # (18:30) the ear's relay settled again on the other voice's own frames (no labels), the kinds as they are
        def evenness(us):
            X = np.concatenate([unit((unit(frames(u)[0]) - relay.mu) @ relay.Z)[::5] for u in us])
            d = np.clip(np.linalg.eigvalsh(np.cov(X.T.astype(np.float64))), 0, None)
            return d.sum() ** 2 / (d * d).sum()
        mine = [utts[i] for i in rng.choice(len(utts), 300, replace=False)]
        pool2 = [u for u in all_utts if u["speaker"] == who2 and u["image"] not in {v["image"] for v in other_u}]
        theirs = [pool2[i] for i in rng.choice(len(pool2), 400, replace=False)]
        say(f"the hearing relay's code, participation ratio of {relay.Z.shape[0]}: its own voice {evenness(mine):.0f}, the other voice {evenness(theirs[:300]):.0f}")
        relay = Relay(relay.shrink).settle(np.concatenate([unit(frames(u)[0])[::5] for u in theirs]))
        say(f"   the other voice through a relay settled on its own frames: {evenness(theirs[:300]):.0f}")
    B_other, V_other = np.stack([bag(u, False) for u in other_u]), see([u["image"] for u in other_u])
    hub = lambda B: (np.sort(B.sum(0))[::-1][:20].sum() / B.sum(), int((B.sum(0) > 0).sum()))  # noqa: E731
    say("kinds used by the test captions (share of all stretches in the 20 most used kinds, kinds used at all): its own voice %.2f, %d; the other voice %.2f, %d" % (hub(B_test) + hub(B_other)))
    M = Bc.T @ V_assoc / len(Bc)
    vocab = {}
    for u in assoc_u:
        for w in set(u["text"].split()):
            vocab[w] = vocab.get(w, 0) + 1
    vocab = {w: k for k, w in enumerate(sorted(w for w, n in vocab.items() if n >= 5))}

    def words(us):
        W = np.zeros((len(us), len(vocab)), np.float32)
        for i, u in enumerate(us):
            for w in u["text"].split():
                if w in vocab:
                    W[i, vocab[w]] += 1
        return W
    Wa = words(assoc_u)
    mw = Wa.mean(0)
    G = (Bc.T @ Bc).astype(np.float64)
    R = np.linalg.solve(G + 1.0 * (np.trace(G) / len(G)) * np.eye(len(G)), Bc.T @ (Wa - mw)).astype(np.float32)
    rec = lambda r: " ".join(f"R@{k} {np.mean(r < k):.3f}" for k in (1, 5, 10))  # noqa: E731
    for name, B, V, us in (("its own voice", B_test, V_test, test_u), (f"a voice never heard ({OTHER})", B_other, V_other, other_u)):
        Bn = unit(B - mu_b)
        S = unit(Bn @ M) @ V.T
        r_img = (S > np.diag(S)[None, :]).sum(0)
        Wt = words(us)
        P = Bn @ R
        Sw = unit(P) @ unit(Wt - mw).T
        r_w = (Sw > np.diag(Sw)[:, None]).sum(1)
        n = (Wt > 0).sum(0)
        cor = lambda lo, hi: np.median([np.corrcoef(P[:, k], Wt[:, k] > 0)[0, 1] for k in np.nonzero((n >= lo) & (n < hi))[0]])  # noqa: E731
        say(f"{name}: photograph -> its caption among {len(us)}: {rec(r_img)}, median rank {np.median(r_img) + 1:.0f} | the caption's own written words from its bag: {rec(r_w)}, median rank {np.median(r_w) + 1:.0f}; "
            f"words heard (correlation median), said in 30+ / 15-29 / 8-14 of the test captions: {cor(30, 10**6):.2f} / {cor(15, 30):.2f} / {cor(8, 15):.2f}")
    if LINK2:
        B2 = np.stack([bag(u, False) for u in link_u])
        V2 = see([u["image"] for u in link_u])
        mu2 = B2.mean(0)
        Bo = unit(B_other - mu2)
        for n in sorted({min(k, len(B2)) for k in (1000, 2500, LINK2)}):
            M2 = unit(B2[:n] - mu2).T @ V2[:n] / n
            for name, Mx, Bx in (("the link learned in the other voice only", M2, Bo), ("that link added to its own voice's", M2 * n + M * len(Bc), Bo)):
                S = unit(Bx @ Mx) @ V_other.T
                r_img, r_cap = (S > np.diag(S)[None, :]).sum(0), (S > np.diag(S)[:, None]).sum(1)
                say(f"this ear, {n} pairs heard in the other voice ({OTHER}), {name}: photograph -> its caption among {len(other_u)}: {rec(r_img)}; caption -> its photograph: {rec(r_cap)}; median rank {np.median(r_img) + 1:.0f} / {np.median(r_cap) + 1:.0f}")
