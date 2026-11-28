"""How much sound does the hearing relay's circuit need, and what happens when the voice changes? (2026-11-28 03:25)

The relay as a circuit of interneurons with its local rule (organic/cortex2.py RelayCircuit, as RELAY=circuit in
experiments/grounded_online_probe.py: unit frames, one block a caption) is born random and hears one reader's
captions in order, bands against the sound so far (fading over TAU). At each count of captions: how evenly its code
is spread on captions it never hears (participation ratio, as the probe prints it). Then the same circuit, as it
ended, hears another reader: the evenness on that reader's held-out captions before and along. Beside them the
relay fitted in a batch on the first reader (the kept ear's), on both voices. The hearing cortex is the kept ear's.

    READER=top2 OTHER=top TAU=0.5 python experiments/relay_circuit_settle_check.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import RelayCircuit  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
READER, OTHER, TAU = os.environ.get("READER", "top2"), os.environ.get("OTHER", "top"), float(os.environ.get("TAU", 0.5))
N, N2, HELD = int(os.environ.get("N", 2000)), int(os.environ.get("N2", 1000)), int(os.environ.get("HELD", 200))
AT = [0, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
KINDS, BANDS = 128, 8
ear = pickle.load(open(os.environ.get("EAR", f"out/bench/grounded_online_ear_s0_r{READER}_live{TAU:g}.pkl"), "rb"))
layer, batch = ear["layer"], ear["relay"]
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
order = sorted(count, key=count.get, reverse=True)
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)  # noqa: E731
t0, k = time.time(), 0.01 / TAU


def say(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def of(reader):
    return [u for u in utts if u["speaker"] == order[int(reader[3:] or 1) - 1]]


class Voice:
    """One reader's captions in order as the hearing cortex's unit frames, each band against the sound so far."""

    def __init__(self):
        self.state = None

    def __call__(self, u):
        g = np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)
        mu0, var0 = (g[0], np.ones(64, np.float32)) if self.state is None else self.state
        mu = lfilter([k], [1, k - 1], g, axis=0, zi=((1 - k) * mu0)[None])[0]
        var = lfilter([k], [1, k - 1], (g - mu) ** 2, axis=0, zi=((1 - k) * var0)[None])[0]
        self.state = (mu[-1].astype(np.float32), var[-1].astype(np.float32))
        r = layer.respond(((g - mu) / (np.sqrt(var) + 1e-3)).astype(np.float32)[..., None], pooled=False)
        b = r.shape[1] // BANDS
        return unit(r[:, : BANDS * b].reshape(r.shape[0], BANDS, b, KINDS).mean(axis=2).reshape(r.shape[0], -1))


def even(C):
    d = np.clip(np.linalg.eigvalsh(np.cov(C.T.astype(np.float64))), 0, None)
    return d.sum() ** 2 / (d * d).sum()


def held(reader):
    v = Voice()
    return np.concatenate([v(u)[::5] for u in of(reader)[-HELD:]])


H1, H2 = held(READER), held(OTHER)
say(f"held out: {len(H1)} and {len(H2)} frames of the last {HELD} captions of readers {READER} and {OTHER}")
say(f"frames as the hearing cortex gives them: {even(H1):.0f} / {even(H2):.0f} of {H1.shape[1]} ({READER} / {OTHER})")
say(f"the relay fitted in a batch on {READER}: {even(unit((H1 - batch.mu) @ batch.Z)):.0f} / {even(unit((H2 - batch.mu) @ batch.Z)):.0f}")

relay, voice, rows = RelayCircuit(KINDS * BANDS, rate=1e-4, block=50, seed=0), Voice(), 0
for n, u in enumerate([None] + of(READER)[:N]):
    if u is not None:
        X = voice(u)
        relay.learn(X)
        rows += len(X)
    if n in AT:
        say(f"circuit after {n} captions of {READER} ({rows / 100:.0f} s of sound): {even(unit(relay.code(H1))):.0f} on {READER}, {even(unit(relay.code(H2))):.0f} on {OTHER}")

voice, rows = Voice(), 0
for n, u in enumerate(of(OTHER)[:N2], 1):
    X = voice(u)
    relay.learn(X)
    rows += len(X)
    if n in AT:
        say(f"then after {n} captions of {OTHER} ({rows / 100:.0f} s of sound): {even(unit(relay.code(H2))):.0f} on {OTHER}, {even(unit(relay.code(H1))):.0f} on {READER}")

own, voice = RelayCircuit(KINDS * BANDS, rate=1e-4, block=50, seed=0), Voice()
for u in of(OTHER)[:N2]:
    own.learn(voice(u))
say(f"a circuit that heard only {OTHER} ({N2} captions): {even(unit(own.code(H2))):.0f} on {OTHER}")
