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
    live.read("h")         # typed letters as they are typed: runs of 3 and 4 letters are the letters organ's units
    live.run()             # until live.stop is set

Events (dicts with "kind"):
    gaze    every 50 ms: where the eye is in the frame (x, y, size as fractions), whether it jumped since the last, and
            what the cochlea gave in those 50 ms (64 bands, bytes)
    recall  when a sound kind is heard: the past view (a small JPEG of what the eye was on, kept in memory only) that
            is most like what the eye held with the kinds now sounding, and how far it stands out among the views kept
    voice   a gesture of the mouth: its number, its two formants, and the sound it makes (16 kHz, 16-bit, 200 ms)
    state   every second: the brain's age, how it keeps up with the clock, the gestures' habits and how many times
            each was heard back, the sight-and-sound link's size; "asleep" when nothing has come in for a while

With nothing coming in for `sleep_after` seconds the brain sleeps: it is kept on disk and its clock stops until a
frame or a sound comes (its age is time lived with a world, and an idle page costs nothing).

The brain is kept on disk (out/live/, never in git) every few minutes and when it stops, and wakes from there.
"""
import collections, os, pickle, threading, time
import numpy as np
from .cochlea import Cochlea, HOP, WINDOW
from .hear_stream import StreamEar, unit
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
        self.cortex, self.code, self.want, self.took, self.seen = cortex, None, None, [], None
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
                self.seen = (w, self.code)
                self.took.append(time.monotonic() - t0)


class LiveBrain:
    def __init__(self, seed=0, home="out/live", emit=None, view=64, cortex_path=None, keep_every=300.0, settings=None, sleep_after=3.0):
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
        if getattr(self.brain, "room", None) is None:  # a microphone is never silent (organic/stream.py, `room`)
            self.brain.room, self.brain.floor, self.brain.top = (10.0, 600.0), None, None
        if getattr(self.brain, "kinds_ear", None) is None:
            # The kinds' ear (organic/hear_stream.py) is born with the brain, all of it: nothing in it was developed on
            # anybody's speech before the life. (2026-11-28 04:30) The relay is a circuit born random (organic/cortex2.py
            # RelayCircuit), learning by its local rule from the frames the ear holds to be sound, a change every 3.3 s
            # of sound worked out beside the clock; at rest after about three minutes of speech (docs/VISION.md
            # 2026-11-28 03:45, 04:30). (2026-11-28 08:55) The hearing cortex under it too (HearingLayer: its own
            # circuit and its kinds' own rule, from the same frames; docs/VISION.md 2026-11-28 07:51, 08:55). Until
            # then it was a SpokenCOCO reader's, kept in ear_born.pkl; that file is no longer read.
            from .cortex2 import RelayCircuit
            from .hear_stream import HearingLayer
            ear = StreamEar(HearingLayer(seed=seed), RelayCircuit(1024, rate=1e-4, block=50, seed=seed), tau=0.5)
            ear.relay_learns, ear.relay_later, ear.relay_block, ear.layer_learns = "sound", True, 330, "sound"
            self.brain.give_ear(ear, kind_tau=1.0)
        if getattr(self.brain, "letters", None) is None:
            self.brain.give_letters()
        self.views, self.view_at, self.recall_at, self.last_seen = collections.deque(maxlen=300), 0.0, 0.0, None  # what the eye was on, for showing a recall; never on disk
        self.ear, self.lock, self.stop = LiveEar(), threading.Lock(), threading.Event()
        self.frame, self.rows, self.texts = None, collections.deque(), collections.deque()
        self.count = collections.Counter()
        self.sleep_after, self.last_in, self.awake = sleep_after, time.monotonic(), threading.Event()

    # --- what comes in, from any thread
    def see(self, img):
        img = np.asarray(img)
        img = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img.astype(np.float32)
        with self.lock:
            self.frame = img
            self.count["frames"] += 1
        self.last_in = time.monotonic()
        self.awake.set()

    def hear(self, samples):
        rows = self.ear.hear(samples)
        with self.lock:
            self.rows.extend(rows)
            self.count["rows"] += len(rows)
            while len(self.rows) > 100:  # more than a second behind: the oldest sound is lost
                self.rows.popleft()
                self.count["rows lost"] += 1
        self.last_in = time.monotonic()
        self.awake.set()

    def read(self, text):
        with self.lock:
            self.texts.extend(text.encode("utf-8", "replace"))
            self.count["letters"] += len(text)
        self.last_in = time.monotonic()
        self.awake.set()

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
        return {"kind": "state", "age": round(b.t, 2), "step_ms": round(1000 * float(np.mean(took)), 2), "step_ms_max": round(1000 * float(np.max(took)), 2),
                "behind_ms": round(1000 * behind), "sight_ms": round(1000 * float(np.mean(self.sight.took)), 1) if self.sight.took else 0.0, "views": len(self.sight.took),
                "frames": n["frames"], "rows": n["rows"], "rows_lost": n["rows lost"], "letters": n["letters"],
                "saccades": n["saccades"], "voices": n["voices"], "links": int(b.n_assoc),
                "link_size": round(float(np.abs(b.link_sum[::8, ::8]).mean() / max(b.n_assoc, 1)), 6) if b.link_sum is not None else 0.0,
                "floor": round(float(b.floor), 2) if b.floor is not None else None, "top": round(float(b.top), 2) if b.top is not None else None,
                "kinds_founded": (b.kinds_ear.pairs.founded + b.kinds_ear.triples.founded) if getattr(b, "kinds_ear", None) is not None else 0,
                "kinds_heard": n["kinds"], "views_kept": len(self.views), "gestures": gestures,
                "relay_heard": float(getattr(getattr(getattr(b, "kinds_ear", None), "relay", None), "n", 0)) / 100,
                "cortex_heard": float(getattr(getattr(getattr(getattr(b, "kinds_ear", None), "layer", None), "circuit", None), "n", 0)) / 100}

    def _recall(self, letters=False):
        """The kept view most like what the eye held with the kinds now sounding (or the letters now typed); how far it
        stands out among them."""
        b = self.brain
        r = b.letters.recall() if letters else b.recall()
        if r is None or len(self.views) < 8:
            return
        codes = np.stack([c for c, _ in self.views])
        codes = codes - codes.mean(0)
        like = unit(codes) @ unit(r)
        best = int(np.argmax(like))
        self.emit({"kind": "recall", "jpeg": self.views[best][1], "likeness": float(like[best]), "stands_out": float((like[best] - like.mean()) / (like.std() + 1e-8)), "letters": letters})

    def _keep_view(self):
        import io
        from PIL import Image
        w, code = self.sight.seen
        out = io.BytesIO()
        Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).save(out, "JPEG", quality=70)
        self.views.append((code, out.getvalue()))

    def run(self, seconds=None):
        b, took, self._heard, jumped = self.brain, [], [], False
        start = nxt = time.monotonic()
        kept, steps = start, 0
        while not self.stop.is_set() and (seconds is None or time.monotonic() - start < seconds):
            if time.monotonic() - self.last_in > self.sleep_after:  # nothing comes in: sleep until something does
                self.keep()
                self.emit({"kind": "state", "asleep": True, "age": round(b.t, 2)})
                self.awake.clear()
                while not self.stop.is_set() and not self.awake.wait(1.0):
                    pass
                nxt = kept = time.monotonic()
                with self.lock:
                    self.rows.clear()
                continue
            with self.lock:
                frame, self.frame = self.frame, None
                row = self.rows.popleft() if self.rows else None
                waiting = len(self.rows)
                typed, self.texts = bytes(self.texts), collections.deque()
            t0 = time.monotonic()
            events = b.step(frame, row)
            for byte in typed:
                b.type(byte, events)
            took.append(time.monotonic() - t0)
            steps += 1
            if row is not None:
                self._heard.append(row)
            for e in events:
                if e[0] == "saccade":
                    jumped = True
                    self.count["saccades"] += 1
                elif e[0] == "kind":
                    self.count["kinds"] += 1
                    if time.monotonic() - self.recall_at > 0.5:
                        self.recall_at = time.monotonic()
                        self._recall()
                elif e[0] == "letters":
                    if time.monotonic() - self.recall_at > 0.5:
                        self.recall_at = time.monotonic()
                        self._recall(letters=True)
                elif e[0] == "sound":
                    self.count["voices"] += 1
                    f1, f2 = (b.gestures[e[1]] if b.gestures is not None else (0, 0))
                    pcm = (np.clip(b.said_pcm, -1, 1) * 32767).astype("<i2").tobytes() if b.said_pcm is not None else b""
                    self.emit({"kind": "voice", "gesture": int(e[1]), "f1": float(f1), "f2": float(f2), "pcm": pcm})
            if steps % 5 == 0:
                if self.sight.seen is not None and self.sight.seen is not self.last_seen and time.monotonic() - self.view_at > 0.5:
                    self.last_seen, self.view_at = self.sight.seen, time.monotonic()
                    self._keep_view()
                g = self._gaze() or (0.25, 0.25, 0.5)
                heard = bytes(64)
                if self._heard:  # for the page's picture only: each band between its own quietest and loudest of the last while
                    h = np.mean(self._heard, axis=0)
                    lo, hi = getattr(self, "_band", (h, h + 1e-3))
                    lo, hi = np.minimum(h, lo + 0.005 * (h - lo)), np.maximum(h, hi + 0.002 * (lo - hi))
                    self._band = (lo, hi)
                    heard = np.clip(255 * (h - lo) / (hi - lo + 1e-3), 0, 255).astype(np.uint8).tobytes()
                self.emit({"kind": "gaze", "x": g[0], "y": g[1], "size": g[2], "jumped": jumped, "heard": heard, "seeing": b.frame is not None})
                jumped, self._heard = False, []
            now = time.monotonic()
            if steps % 100 == 0:
                self.emit(self._state(took, max(0.0, now - nxt)))
                took, b.log, self.sight.took = [], [], []
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
