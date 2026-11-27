"""Are two voices' sound kinds the same words, seen through sight? (docs/VISION.md 2026-11-27 18:34, said before.)

Two ears that each grew up in one voice (experiments/grounded_speech_probe.py, READER=top / top2 / top3) share no
kinds: a one-voice ear hears almost nothing in another voice (18:30). But each has a Hebb link from its kinds to the
same sight. Here, from the two runs' saved bags and photograph codes: each kind's sight side (the link's row: the
mean photograph code of the captions it is heard in, centred), and for each kind of reader A the kind of reader B
whose sight side is most like it. Scored by the written words (a measure only): a kind's word is the commonest
written word of its captions; a kind is word-like if that word is in half its captions or more and in a fifth of
all captions or fewer (as on 2026-11-26). Does the matched kind carry the same word?

    A=top B=top2 python experiments/voices_through_sight_check.py
"""
import json, os
import numpy as np

DIR = os.path.expanduser("~/datasets/spokencoco")
A, B, SEED = os.environ.get("A", "top"), os.environ.get("B", "top2"), int(os.environ.get("SEED", 0))
EAR_U, ASSOC_U, TEST, DEV = 4670, 5000, 1000, 16000
all_utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in all_utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
ranked = sorted(count, key=count.get, reverse=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def reader(name):
    """The run's own captions for the link (the same calls on the same random state), its bags, its photograph codes."""
    who = ranked[int(name[3:] or 1) - 1]
    utts = [u for u in all_utts if u["speaker"] == who]
    rng = np.random.default_rng(SEED)
    images = sorted({u["image"] for u in utts})
    images = [images[i] for i in rng.permutation(len(images))]
    test_imgs, assoc_imgs = images[:TEST], images[TEST: TEST + ASSOC_U]
    by_image = {}
    for u in utts:
        by_image.setdefault(u["image"], []).append(u)
    [by_image[i][int(rng.integers(len(by_image[i])))] for i in test_imgs]
    assoc_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in assoc_imgs]
    bags = np.load(f"out/bench/grounded_bags_s{SEED}_e{EAR_U}_a{ASSOC_U}_t{TEST}_d{DEV}_r{name}.npz")["assoc"]
    V = np.load(f"out/bench/grounded_codes_s{SEED}_a{ASSOC_U}_t{TEST}_r{name}.npz")["assoc"].astype(np.float32)
    words = [set(u["text"].split()) for u in assoc_u]
    base = {}
    for ws in words:
        for w in ws:
            base[w] = base.get(w, 0) + 1
    label, like, n_caps = [], [], []
    for k in range(bags.shape[1]):
        caps = np.nonzero(bags[:, k] > 0)[0]
        c = {}
        for i in caps:
            for w in words[i]:
                c[w] = c.get(w, 0) + 1
        ok = {w: n for w, n in c.items() if base[w] <= 0.2 * len(words)}
        w = max(ok, key=ok.get) if ok else ""
        label.append(w)
        like.append(len(caps) >= 10 and bool(ok) and ok[w] >= 0.5 * len(caps))
        n_caps.append(len(caps))
    M = unit(bags - bags.mean(0)).T @ (V - V.mean(0)) / len(bags)  # Hebb, as the run's; the sight side of each kind
    return unit(M), np.array(label), np.array(like), np.array(n_caps), assoc_imgs


MA, LA, KA, NA, IA = reader(A)
MB, LB, KB, NB, IB = reader(B)
print(f"{A}: {int(KA.sum())} word-like kinds of {len(KA)} ({len(set(LA[KA]))} different words); {B}: {int(KB.sum())} ({len(set(LB[KB]))}); photographs both readers' links saw: {len(set(IA) & set(IB))} of {len(IA)}")
rng = np.random.default_rng(SEED + 5)
for name, cand in (("any kind of the other reader", NB >= 10), ("the other reader's word-like kinds only", KB)):
    idx = np.nonzero(cand)[0]
    S = MA[KA] @ MB[idx].T
    hit = LB[idx[np.argmax(S, axis=1)]] == LA[KA]
    top5 = np.array([LA[KA][i] in set(LB[idx[np.argsort(-S[i])[:5]]]) for i in range(len(S))])
    chance = np.mean([np.mean(LB[rng.choice(idx, int(KA.sum()))] == LA[KA]) for _ in range(200)])
    has = np.isin(LA[KA], LB[idx])
    words = sorted({w for w, h in zip(LA[KA], hit) if h})
    print(f"{A}'s word-like kinds matched through sight to {name} ({len(idx)}): the same written word {hit.mean():.3f} (by chance {chance:.3f}); in the five most alike {top5.mean():.3f}; "
          f"where the other reader has a kind with that word at all ({has.mean():.2f} of them): {hit[has].mean():.3f} | words matched: " + ", ".join(w.lower() for w in words[:40]))
