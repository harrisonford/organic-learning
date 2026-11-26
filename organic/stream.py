"""A stream brain: no turns, a 10 ms clock, streams in and streams out at once (docs/ASYNC.md; Kris, 2026-11-26).

Kris: "every neuron ticks and send info ... stream in stream out". Every population here updates at each step of
the clock (DT, 10 ms) with its own time constant, and passes on only what changed (as spikes carry only events);
every output is a population racing to its own threshold, and whichever wins acts, while everything else goes on.
Nothing waits for anything: the eye moves when its drive wins, the mouth sounds when its drive wins, hearing and
seeing never stop.

This first version is the substrate, not a learner: the eye (reflex salience and a race between places), hearing
(the cochlea's frames through a hearing cortex built by the eye's own rule) and a babbling mouth, with one plastic
link (sight and sound associated by Hebb's rule while both stream). What it is for: to show the parts running at
once and to measure their rhythms and their cost, before the learned organs move in (docs/ASYNC.md).

    brain = StreamBrain(cortex, hearing_layer)
    for t, frame, sound_row in world:        # frame or None (a new picture), sound_row or None (10 ms of sound)
        events = brain.step(frame, sound_row) # what the brain did in these 10 ms: looked, sounded
"""
import numpy as np

DT = 0.010  # seconds a step


class Leaky:
    """A population's state: it follows its input with time constant tau (seconds); `changed` says whether it moved
    enough to be worth sending on (change-only propagation)."""

    def __init__(self, n, tau, eps=1e-3):
        self.x, self.k, self.eps, self.sent = np.zeros(n, np.float32), DT / tau, eps, np.zeros(n, np.float32)

    def step(self, drive):
        self.x += self.k * (drive - self.x)
        if np.abs(self.x - self.sent).max() > self.eps * (np.abs(self.sent).max() + 1e-6):
            self.sent = self.x.copy()
            return True
        return False


class Race:
    """Outputs racing to a threshold of 1: each accumulates its drive plus noise; the first to cross acts, and all
    start again (the race of movement populations before a saccade; a LATER-like accumulator). The drives are taken
    relative to the strongest (divisive normalisation), times `rate`: the strongest crosses in about 1 / rate seconds,
    the rate the act is known to have (3-4 fixations a second); noise lets a weaker one win at times."""

    def __init__(self, n, rate, noise, rng):
        self.acc, self.rate, self.noise, self.rng = np.zeros(n, np.float32), rate, noise, rng

    def step(self, drive):
        top = float(drive.max())
        d = self.rate * drive / top if top > 0 else np.zeros_like(drive)
        self.acc += DT * d + self.noise * np.sqrt(DT) * self.rng.standard_normal(len(d)) * (drive > 0)
        self.acc = np.maximum(self.acc, 0)
        k = int(np.argmax(self.acc))
        if self.acc[k] >= 1.0:
            self.acc[:] = 0
            return k
        return None


def places(img, side=None):
    """The nine places the eye can point to (3 x 3 windows of half the view) and how far each stands out from its
    surround (organic/gaze.py's centre-surround, every place kept)."""
    img = np.asarray(img, np.float32)
    H, W = img.shape[:2]
    side = side or H // 2
    ring, step = side // 8, side // 2
    energy = np.abs(img - np.median(img.reshape(-1, img.shape[2]), axis=0)).mean(axis=2)
    total = np.pad(np.pad(energy, ring).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    area = lambda r0, c0, r1, c1: total[r1, c1] - total[r0, c1] - total[r1, c0] + total[r0, c0]  # noqa: E731
    out = []
    for r in range(0, H - side + 1, step):
        for c in range(0, W - side + 1, step):
            inside = area(r + ring, c + ring, r + ring + side, c + ring + side)
            outside = area(r, c, r + side + 2 * ring, c + side + 2 * ring) - inside
            r0, c0, r1, c1 = max(r - ring, 0), max(c - ring, 0), min(r + side + ring, H), min(c + side + ring, W)
            ring_area = max((r1 - r0) * (c1 - c0) - side * side, 1)
            out.append((r, c, max(inside / (side * side) - outside / ring_area, 0.0)))
    return out


class StreamBrain:
    def __init__(self, cortex, hearing, seed=0, motor_units=32, fixation_rate=3.5, babble_rate=2.0):
        self.rng = np.random.default_rng(seed)
        self.cortex, self.hearing = cortex, hearing  # a VisualCortex2 with its relay; a Layer2 on the cochlea's frames
        self.t, self.frame, self.places, self.where = 0.0, None, [], 4  # the centre place to begin with
        self.ior = np.zeros(9, np.float32)  # inhibition of return, fading over about a second
        self.eye = Race(9, fixation_rate, 0.5, self.rng)
        self.blind = 0.0  # a saccade takes about 30 ms, in which nothing new is seen
        self.sight = Leaky(cortex.relay.Z.shape[0], tau=0.05)  # what is seen, settling over 50 ms after each change
        self.sound_rows = []  # the last 80 ms of the cochlea, for the hearing cortex's window
        self.sound = None  # the hearing population, sized at the first sound
        self.mouth = Race(motor_units, babble_rate, 0.5, self.rng)  # spontaneous babble, a few a second
        self.habit = np.ones(motor_units, np.float32)
        self.assoc = None  # sight <-> sound, Hebb's rule while both stream (experiments/av_probe.py)
        self.n_assoc = 0
        self.log = []

    def _see(self):
        """The code of what the eye is on: the fixated half of the frame, enlarged to fill the view."""
        from PIL import Image
        r, c, _ = self.places[self.where]
        side = self.frame.shape[0] // 2
        w = self.frame[r: r + side, c: c + side]
        w = np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize(self.frame.shape[:2][::-1], Image.BILINEAR), np.float32) / 255
        return self.cortex.whole(w, 4)

    def _hear(self, row):
        self.sound_rows.append(np.asarray(row, np.float32))
        if len(self.sound_rows) > 8:
            self.sound_rows.pop(0)
        if len(self.sound_rows) < 8:
            return None
        r = self.hearing.respond(np.stack(self.sound_rows)[..., None], pooled=False)  # (1, bands, kinds)
        b = r.shape[1] // 4
        v = r[0, : 4 * b].reshape(4, b, -1).mean(axis=1).ravel()
        return v / (np.linalg.norm(v) + 1e-8)

    def step(self, frame=None, sound_row=None):
        """10 ms of life. Returns the events of this step: ('saccade', place), ('sound', unit)."""
        self.t += DT
        events = []
        if frame is not None:  # a new picture: the places that stand out are measured again
            self.frame = np.asarray(frame, np.float32)
            self.places = places(self.frame)
            self.view = self._see()
        if self.frame is not None:
            self.ior *= np.exp(-DT / 1.0)
            sal = np.array([s for _, _, s in self.places], np.float32)
            drive = sal / (sal.sum() + 1e-8) * (1 - self.ior)
            drive[self.where] = 0.0  # a saccade goes elsewhere
            k = self.eye.step(drive) if self.blind <= 0 else None
            if k is not None:
                self.ior[self.where] = 1.0
                self.where, self.blind = k, 0.03
                events.append(("saccade", k))
            self.blind -= DT
            if self.blind <= 0 and (k is not None or frame is not None):
                self.view = self._see()
            moved = self.sight.step(self.view if self.blind <= 0 else 0 * self.view)
        else:
            moved = False
        if sound_row is not None:
            h = self._hear(sound_row)
            if h is not None:
                if self.sound is None:
                    self.sound = Leaky(len(h), tau=0.05)
                    self.assoc = np.zeros((len(h), len(self.sight.x)), np.float32)
                heard = self.sound.step(h)
                self.steps_since_link = getattr(self, "steps_since_link", 0) + 1
                # Hebb's rule while both stream (the running mean of their product), at the pace the two populations
                # change (their 50 ms time constant), not every 10 ms
                if (heard or moved) and self.frame is not None and self.steps_since_link >= 5:
                    self.steps_since_link = 0
                    self.n_assoc += 1
                    self.assoc += (np.outer(self.sound.x, self.sight.x) - self.assoc) / self.n_assoc
        u = self.mouth.step(self.habit)
        if u is not None:
            self.habit[u] += 1  # a gesture made becomes a little more likely (habit)
            events.append(("sound", u))
        for e in events:
            self.log.append((round(self.t, 3),) + e)
        return events
