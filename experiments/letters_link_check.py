"""Typed letters as a stream: runs of letters hashed to units, tied to what is in view (docs/VISION.md 2026-11-28 01:28).

A caption as written, letter by letter: every run of N letters (spaces included) excites one of UNITS units (a
fixed hash, as organic/ear.py's combination neurons); a caption's units are tied to its photograph's code by
Hebb's rule in one pass, as the one-pass ear ties a bag (less its running mean, at unit length) or plain. Read on
the held-out pairs. Beside it: one unit a written word. Uses the photographs' codes kept by
experiments/grounded_link_check.py for the same reader, order and test.

    READER=top2 python experiments/letters_link_check.py
"""
import json, os, sys, zlib
import numpy as np

DIR = os.path.expanduser("~/datasets/spokencoco")
READER, SEED, TEST, UNITS = os.environ.get("READER", "top2"), int(os.environ.get("SEED", 0)), int(os.environ.get("TEST", 1000)), int(os.environ.get("UNITS", 2048))
CACHE = os.environ.get("CACHE", f"out/bench/grounded_online_ear_s{SEED}_r{READER}_live2_heard.npz")
unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)  # noqa: E731

rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1]
utts = [u for u in utts if u["speaker"] == who]
by_image = {}
for u in utts:
    by_image.setdefault(u["image"], []).append(u)
images = sorted(by_image)
images = [images[i] for i in rng.permutation(len(images))]
test_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[:TEST]]
life_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[TEST:]]
z = np.load(CACHE)
V_life, V_test = z["V_life"], z["V_test"]
assert len(V_life) == len(life_u) and len(V_test) == len(test_u)
rec = lambda r: " ".join(f"R@{k} {np.mean(r < k):.3f}" for k in (1, 5, 10))  # noqa: E731


def read(X, Xt, normed, name):
    mean, A = np.zeros(X.shape[1], np.float64), np.empty_like(X)
    for n in range(len(X)):
        mean += (X[n] - mean) / (n + 1)
        A[n] = unit(X[n] - mean.astype(np.float32)) if normed else X[n]
    M = A.T @ V_life
    S = unit(unit(Xt - mean.astype(np.float32)) @ M) @ V_test.T
    r_img = (S > np.diag(S)[None, :]).sum(0)
    print(f"{name}, {'the mean taken' if normed else 'plain'}: photograph -> its caption among {len(Xt)}: {rec(r_img)}; median rank {np.median(r_img) + 1:.0f}", flush=True)


def runs(us, ns):
    X = np.zeros((len(us), UNITS), np.float32)
    for i, u in enumerate(us):
        t = (" " + u["text"].lower().strip() + " ").encode()
        for n in ns:
            for a in range(len(t) - n + 1):
                X[i, zlib.crc32(t[a: a + n]) % UNITS] += 1
    return X


vocab = {}
for u in life_u:
    for w in u["text"].lower().split():
        vocab.setdefault(w, len(vocab))


def words(us):
    X = np.zeros((len(us), len(vocab)), np.float32)
    for i, u in enumerate(us):
        for w in u["text"].lower().split():
            if w in vocab:
                X[i, vocab[w]] += 1
    return X


print(f"reader {READER}: {len(life_u)} captions as written with their photographs, {len(test_u)} held out; {len(vocab)} different words; {np.mean([len(u['text']) for u in life_u]):.0f} letters a caption")
for normed in (True, False):
    read(words(life_u), words(test_u), normed, "one unit a written word")
    for ns in ((2,), (3,), (4,), (2, 3), (3, 4), (2, 3, 4)):
        read(runs(life_u, ns), runs(test_u, ns), normed, f"runs of {' and '.join(map(str, ns))} letters, {UNITS} units")
