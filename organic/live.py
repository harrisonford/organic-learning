"""The stream brain on the wall clock: a camera and a microphone in, where it looks and what it babbles out
(docs/LIVE.md; Kris, 2026-11-27: a live page for kid-a, "hints of something forming").

`organic/stream.py` is a brain of populations on a 10 ms clock that so far lived on recorded video and sound, as fast
as the machine went. Here the same brain, unchanged, runs against real time: every 10 ms it takes the newest camera
frame if one came (older ones are skipped: an eye does not queue what it missed) and the next 10 ms of sound if
there is any, and what it does goes out as events. Nothing here knows how the frames arrive or where the events go
(a database, a socket, a test): `see`, `hear` and `read` may be called from any thread, and `emit` is called from
the brain's own.

    live = LiveBrain(seed=0, emit=print)
    live.see(frame)        # H x W x 3, floats in [0, 1] or bytes 0..255; square
    live.hear(samples)     # mono, 16 kHz, floats in [-1, 1]; any length
    live.read("hello")     # typed text: kept and counted, no organ takes it yet (docs/LIVE.md)
    live.run()             # until live.stop is set

Events (dicts with "kind"):
    gaze    every 50 ms: where the eye is in the frame (x, y, size as fractions), whether it jumped since the last
    voice   a gesture of the mouth: its number, its two formants, and the sound it makes (16 kHz, 16-bit, 200 ms)
    state   every second: the brain's age, how it keeps up with the clock, what it heard (64 bands), the gestures'
            habits and how many times each was heard back, the sight-and-sound link's size

The brain is kept on disk (out/live/, never in git) every few minutes and when it stops, and wakes from there.
"""
import collections, os, pickle, threading, time
import numpy as np
from .cochlea import Cochlea, HOP, WINDOW
from .stream import DT, StreamBrain
from .vocal import VocalTract


class LiveEar:
    """The cochlea on sound that comes in pieces: the samples a frame still needs are kept for the next piece."""

    def __init__(self):
        self.cochlea, self.tail = Cochlea(), np.zeros(0, np.float32)

    def hear(self, samples):
        x = np.concatenate([self.tail, np.asarray(samples, np.float32).ravel()])
        if len(x) < WINDOW:
            self.tail = x
            return []
        rows = self.cochlea.hear(x)
        self.tail = x[len(rows) * HOP:]
        return list(rows)


def hearing_cortex(seed, path, corpus="~/datasets/spokencoco"):
    """The hearing cortex the stream brain is born with: the eye's layer on the cochlea's frames, developed on speech
    before the life (as in experiments/babble_drift_probe.py), once, and kept."""
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    import json
    from .cortex2 import Layer2
    rng, corpus = np.random.default_rng(seed), os.path.expanduser(corpus)
    utts = [u for u in map(json.loads, open(os.path.join(corpus, "cochlea.jsonl"))) if u["split"] == "val" and u["rows"] >= 120]
    grams = np.memmap(os.path.join(corpus, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
    clip = lambda u: (lambda s: np.asarray(grams[s: s + 100], np.float32)[..., None])(u["start"] + int(rng.integers(0, u["rows"] - 100)))  # noqa: E731
    layer = Layer2(1, 128, 8, 1, seed=seed)
    layer.settle((clip(utts[rng.integers(len(utts))]) for _ in range(800)), rng)
    for n in range(2000):
        layer.learn(clip(utts[rng.integers(len(utts))]), rng, rate=0.05 * (1 - n / 2000) + 0.005)
    pickle.dump(layer, open(path, "wb"))
    return layer


class LaterSight:
    """The visual cortex beside the clock: it takes the newest fixated window when it is free (windows that came while
    it was busy are never looked at) and `code` is its answer to the last one it finished."""

    def __init__(self, cortex):
        self.cortex, self.code, self.want, self.took = cortex, None, None, []
        self.wake, self.stop = threading.Event(), False
        threading.Thread(target=self._work, daemon=True).start()

    def __call__(self, window):
        self.want = window
        self.wake.set()

    def _work(self):
        while not self.stop:
            self.wake.wait(0.5)
            self.wake.clear()
            w, self.want = self.want, None
            if w is not None:
                t0 = time.monotonic()
                self.code = self.cortex.whole(w, 4)
                self.took.append(time.monotonic() - t0)


class LiveBrain:
    def __init__(self, seed=0, home="out/live", emit=None, view=64, cortex_path=None, keep_every=300.0, settings=None):
        os.makedirs(home, exist_ok=True)
        self.home, self.seed, self.emit, self.keep_every = home, seed, emit or (lambda e: None), keep_every
        self.cortex = pickle.load(open(cortex_path or f"out/bench/cortex_ph_d_s{seed}.pkl", "rb"))
        self.hearing = hearing_cortex(seed, os.path.join(home, f"hearing_s{seed}.pkl"))
        self.path = os.path.join(home, f"brain_s{seed}.pkl")
        if os.path.exists(self.path):
            self.brain = pickle.load(open(self.path, "rb"))
            self.brain.cortex, self.brain.hearing = self.cortex, self.hearing
        else:
            # the settings the probes ended on: both reflexes of the eye (VISION.md 2026-11-27 04:57), eight stretches of
            # bands, habits from what is heard and none from the mouth's own repetition (babble_drift_probe.py)
            s = dict(bands=8, habit_step=0.0, heard_habit=0.01, motion_drive=True, pursuit_tau=0.1, view=view)
            s.update(settings or {})
            self.brain = StreamBrain(self.cortex, self.hearing, seed=seed, tract=VocalTract(f0=280.0, seed=seed), **s)
        self.brain.sight_later = self.sight = LaterSight(self.cortex)
        self.ear, self.lock, self.stop = LiveEar(), threading.Lock(), threading.Event()
        self.frame, self.rows, self.texts = None, collections.deque(), collections.deque()
        self.count = collections.Counter()

    # --- what comes in, from any thread
    def see(self, img):
        img = np.asarray(img)
        img = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img.astype(np.float32)
        with self.lock:
            self.frame = img
            self.count["frames"] += 1

    def hear(self, samples):
        rows = self.ear.hear(samples)
        with self.lock:
            self.rows.extend(rows)
            self.count["rows"] += len(rows)
            while len(self.rows) > 100:  # more than a second behind: the oldest sound is lost
                self.rows.popleft()
                self.count["rows lost"] += 1

    def read(self, text):
        with self.lock:
            self.texts.append((time.time(), text))
            self.count["letters"] += len(text)

    # --- the brain's own thread
    def keep(self):
        b = self.brain
        b.cortex, b.hearing, log, b.sight_later = None, None, b.log, None
        b.log = []
        try:
            pickle.dump(b, open(self.path + ".tmp", "wb"))
            os.replace(self.path + ".tmp", self.path)
        finally:
            b.cortex, b.hearing, b.log, b.sight_later = self.cortex, self.hearing, log, self.sight

    def _gaze(self):
        b = self.brain
        if b.frame is None:
            return None
        H = b.frame.shape[0]
        r, c = (b.pos if (b.pursuit_tau and b.pos is not None) else b.places[b.where][:2])
        return float(c) / H, float(r) / H, 0.5

    def _state(self, took, behind):
        b, n = self.brain, self.count
        gestures = []
        if b.gestures is not None:
            share = b.habit / b.habit.sum()
            gestures = [[round(float(f1)), round(float(f2)), round(float(h), 4), int(k)] for (f1, f2), h, k in zip(b.gestures, share, b.amap_n)]
        heard = np.mean(self._heard, axis=0) if self._heard else np.zeros(64, np.float32)
        return {"kind": "state", "age": round(b.t, 2), "step_ms": round(1000 * float(np.mean(took)), 2), "step_ms_max": round(1000 * float(np.max(took)), 2),
                "behind_ms": round(1000 * behind), "sight_ms": round(1000 * float(np.mean(self.sight.took)), 1) if self.sight.took else 0.0, "views": len(self.sight.took),
                "frames": n["frames"], "rows": n["rows"], "rows_lost": n["rows lost"], "letters": n["letters"],
                "saccades": n["saccades"], "voices": n["voices"], "links": int(b.n_assoc),
                "link_size": round(float(np.abs(b.link_sum[::8, ::8]).mean() / max(b.n_assoc, 1)), 6) if b.link_sum is not None else 0.0,
                "heard": [int(v) for v in np.clip(heard * 12, 0, 255)], "gestures": gestures}

    def run(self, seconds=None):
        b, took, self._heard, jumped = self.brain, [], [], False
        start = nxt = time.monotonic()
        kept, steps = start, 0
        while not self.stop.is_set() and (seconds is None or time.monotonic() - start < seconds):
            with self.lock:
                frame, self.frame = self.frame, None
                row = self.rows.popleft() if self.rows else None
                waiting = len(self.rows)
            t0 = time.monotonic()
            events = b.step(frame, row)
            took.append(time.monotonic() - t0)
            steps += 1
            if row is not None:
                self._heard.append(row)
            for e in events:
                if e[0] == "saccade":
                    jumped = True
                    self.count["saccades"] += 1
                elif e[0] == "sound":
                    self.count["voices"] += 1
                    f1, f2 = (b.gestures[e[1]] if b.gestures is not None else (0, 0))
                    pcm = (np.clip(b.said_pcm, -1, 1) * 32767).astype("<i2").tobytes() if b.said_pcm is not None else b""
                    self.emit({"kind": "voice", "gesture": int(e[1]), "f1": float(f1), "f2": float(f2), "pcm": pcm})
            if steps % 5 == 0:
                g = self._gaze()
                if g is not None:
                    self.emit({"kind": "gaze", "x": g[0], "y": g[1], "size": g[2], "jumped": jumped})
                jumped = False
            now = time.monotonic()
            if steps % 100 == 0:
                self.emit(self._state(took, max(0.0, now - nxt)))
                took, self._heard, b.log, self.sight.took = [], [], [], []
            if now - kept > self.keep_every:
                self.keep()
                kept = time.monotonic()
            nxt += DT
            now = time.monotonic()
            if now - nxt > 0.5:  # far behind the clock (a stall): the clock is not chased, time was lost
                self.count["stalls"] += 1
                nxt = now
            elif waiting <= 10 and nxt > now:  # sound waiting: catch up without sleeping
                time.sleep(nxt - now)
        self.keep()


if __name__ == "__main__":  # a dry run without camera or network: a moving square and an adult's vowels, for SECONDS
    import sys
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    seen = collections.Counter()
    last = {}

    def emit(e):
        seen[e["kind"]] += 1
        last[e["kind"]] = e

    live = LiveBrain(seed=0, home=os.environ.get("LIVE_HOME", "out/tmp/live_dry"), emit=emit)
    adult, rng = VocalTract(f0=140.0, seed=1), np.random.default_rng(0)

    def world():  # 20 frames and 20 pieces of sound a second; an "i" every 1.5 s
        t0, k, sound = time.monotonic(), 0, np.zeros(0, np.float32)
        while not live.stop.is_set():
            t = time.monotonic() - t0
            img = np.full((128, 128, 3), 0.4, np.float32) + 0.05 * rng.standard_normal((128, 128, 3)).astype(np.float32)
            x = int(48 + 40 * np.sin(t))
            img[40:70, x: x + 30] = (0.9, 0.2, 0.2)
            live.see(np.clip(img, 0, 1))
            if k % 30 == 0:
                sound = np.concatenate([sound, adult.say(300, 2300, 200)])
            piece, sound = sound[:800], sound[800:]
            live.hear(np.concatenate([piece, np.zeros(800 - len(piece), np.float32)]))
            k += 1
            time.sleep(max(0.0, t0 + 0.05 * k - time.monotonic()))

    threading.Thread(target=world, daemon=True).start()
    live.run(seconds)
    live.stop.set()
    s = last.get("state", {})
    print({k: v for k, v in s.items() if k not in ("heard", "gestures")})
    print("events:", dict(seen), "| last gaze:", last.get("gaze"), "| last voice:", {k: (len(v) if k == "pcm" else v) for k, v in last.get("voice", {}).items()})
