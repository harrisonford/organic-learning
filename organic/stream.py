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


def motion(img, before, reach=3):
    """What changed between two frames against the view's own flow: the frame before is shifted by the whole-pixel
    step (up to `reach` each way) that matches it best, and what still changes moved by itself (the self-motion of
    head and camera discounted, as the optokinetic system and MST do; to verify). Each place's change less the
    frame's median change; zero at the edge the shift cannot reach (2026-11-27, experiments/core50_pursuit_probe.py)."""
    a, b = np.asarray(img, np.float32), np.asarray(before, np.float32)
    H, W = a.shape[:2]
    best, m = None, None
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            d = np.abs(a[reach: H - reach, reach: W - reach] - b[reach - dy: H - reach - dy, reach - dx: W - reach - dx]).mean(axis=2)
            if best is None or d.mean() < best:
                best, m = d.mean(), d
    m = np.pad(m, reach)
    return np.maximum(m - np.median(m), 0)


class StreamBrain:
    def __init__(self, cortex, hearing, seed=0, motor_units=32, fixation_rate=3.5, babble_rate=2.0, tract=None, bands=4, habit_step=1.0, echo_tau=None, heard_habit=0.0,
                 motion_drive=False, pursuit_tau=None):
        self.rng = np.random.default_rng(seed)
        # Two reflexes of the eye, off by default (2026-11-27, docs/VISION.md 04:57): `motion_drive`, what moves against
        # the view's own flow draws saccades as well as what stands out (each relative to its own strongest, then
        # added: the colliculus's priority map; to verify); `pursuit_tau` (seconds), between saccades the eye drifts
        # toward the centre of that motion within its own field (smooth pursuit: Lisberger 2010).
        self.motion_drive, self.pursuit_tau = motion_drive, pursuit_tau
        self.before, self.moving, self.pos, self.pos_seen, self.target = None, None, None, None, None
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
        self.habit_step = habit_step  # how much a gesture made becomes more likely (0: babbling explores evenly)
        self.bands = bands  # stretches of the bands the hearing code keeps apart
        # a vocal tract (organic/vocal.py): each motor unit a gesture, a place in the vowel space; what it sounds like is
        # heard, not known. None: the mouth makes symbols, as in the first version.
        self.tract, self.gestures, self.own = tract, None, []
        if tract is not None:
            f1 = self.rng.uniform(250, 850, motor_units)
            f2 = np.maximum(self.rng.uniform(700, 2500, motor_units), f1 + 300)
            self.gestures = np.stack([f1, f2], 1)
        self.saying, self.said_codes = None, []  # the gesture being said and the hearing codes while it sounds
        self.amap, self.amap_n = None, np.zeros(motor_units)  # what each gesture sounds like (Hebb: the running mean of the hearing while it is said)
        self.imitate = tract is not None  # what is heard drives the gestures whose learned sound matches it (covert imitation)
        # an echo of what was heard (echoic memory lasts a second or more; to verify: Cowan 1984): the last stretch of
        # sound, from its onset to the silence after it, held as a whole and fading over echo_tau seconds until the next
        # stretch replaces it (a fading average mixed each sound with the one before: tried 2026-11-26, imitation fell to
        # chance). Not taken in while the mouth sounds (speaking-induced suppression: Houde et al. 2002, to verify).
        # Imitation reads the stretch being heard, or else the echo. None: the 50 ms hearing population, as before.
        self.echo_tau, self.echo, self.echo_strength, self.stretch, self.stretch_n = echo_tau, None, 0.0, None, 0
        self.heard_habit = heard_habit  # habits from what is heard: a heard sound strengthens the habits of the gestures it matches
        self.assoc = None  # sight <-> sound, Hebb's rule while both stream (experiments/av_probe.py)
        self.n_assoc = 0
        self.log = []

    def _see(self):
        """The code of what the eye is on: the fixated half of the frame, enlarged to fill the view."""
        from PIL import Image
        r, c, _ = self.places[self.where]
        if self.pursuit_tau:  # the eye is where pursuit has carried it, not only on one of the nine places
            r, c = int(round(self.pos[0])), int(round(self.pos[1]))
            self.pos_seen = self.pos.copy()
        side = self.frame.shape[0] // 2
        w = self.frame[r: r + side, c: c + side]
        w = np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize(self.frame.shape[:2][::-1], Image.BILINEAR), np.float32) / 255
        return self.cortex.whole(w, 4)

    def _pursuit_target(self):
        """Where pursuit carries the eye: the fixated half centred on the centre of the motion within the eye's own
        field (the half it is on and a quarter of it around), or None when nothing there moves."""
        if self.moving is None:
            return None
        H, W = self.frame.shape[:2]
        side = H // 2
        margin = side // 4
        r0, c0 = int(max(self.pos[0] - margin, 0)), int(max(self.pos[1] - margin, 0))
        r1, c1 = int(min(self.pos[0] + side + margin, H)), int(min(self.pos[1] + side + margin, W))
        field = self.moving[r0:r1, c0:c1]
        if field.sum() <= 1e-6:
            return None
        rr, cc = np.mgrid[r0:r1, c0:c1]
        centre = np.array([(rr * field).sum(), (cc * field).sum()], np.float32) / field.sum()
        return np.clip(centre - side / 2, 0, H - side).astype(np.float32)

    def _hear(self, row):
        self.sound_rows.append(np.asarray(row, np.float32))
        if len(self.sound_rows) > 8:
            self.sound_rows.pop(0)
        if len(self.sound_rows) < 8:
            return None
        r = self.hearing.respond(np.stack(self.sound_rows)[..., None], pooled=False)  # (1, bands, kinds)
        n = self.bands
        b = r.shape[1] // n
        v = r[0, : n * b].reshape(n, b, -1).mean(axis=1).ravel()
        return v / (np.linalg.norm(v) + 1e-8)

    def step(self, frame=None, sound_row=None):
        """10 ms of life. Returns the events of this step: ('saccade', place), ('sound', unit)."""
        self.t += DT
        events = []
        if frame is not None:  # a new picture: the places that stand out are measured again
            self.frame = np.asarray(frame, np.float32)
            self.places = places(self.frame)
            if self.motion_drive or self.pursuit_tau:  # what moved since the frame before, against the view's own flow
                self.moving = motion(self.frame, self.before) if self.before is not None else None
                self.before = self.frame
                if self.moving is not None:
                    self.moving_places = np.array([s for _, _, s in places(np.repeat(self.moving[..., None], 3, axis=2))], np.float32)
                if self.pursuit_tau:
                    if self.pos is None:
                        self.pos = np.array(self.places[self.where][:2], np.float32)
                    self.target = self._pursuit_target()
            self.view = self._see()
        if self.frame is not None:
            self.ior *= np.exp(-DT / 1.0)
            sal = np.array([s for _, _, s in self.places], np.float32)
            if self.motion_drive and self.moving is not None:  # each source relative to its own strongest, then added
                mot = self.moving_places
                drive = (sal / (sal.max() + 1e-8) + mot / (mot.max() + 1e-8)) * (1 - self.ior)
            else:
                drive = sal / (sal.sum() + 1e-8) * (1 - self.ior)
            drive[self.where] = 0.0  # a saccade goes elsewhere
            k = self.eye.step(drive) if self.blind <= 0 else None
            if k is not None:
                self.ior[self.where] = 1.0
                self.where, self.blind = k, 0.03
                events.append(("saccade", k))
                if self.pursuit_tau:
                    self.pos = np.array(self.places[k][:2], np.float32)
                    self.target = self._pursuit_target()
            self.blind -= DT
            drifted = False
            if self.pursuit_tau and self.target is not None and self.blind <= 0:  # pursuit: the eye drifts after the motion
                self.pos += (DT / self.pursuit_tau) * (self.target - self.pos)
                drifted = np.abs(self.pos - self.pos_seen).max() >= 2 if self.pos_seen is not None else True
            if self.blind <= 0 and (k is not None or frame is not None or drifted):
                self.view = self._see()
            moved = self.sight.step(self.view if self.blind <= 0 else 0 * self.view)
        else:
            moved = False
        if self.own:  # the mouth's own sound reaches the ear with the world's (energies add; the cochlea compresses by 0.3)
            own = self.own.pop(0)
            sound_row = own if sound_row is None else (np.asarray(sound_row, np.float32) ** (1 / 0.3) + own ** (1 / 0.3)) ** 0.3
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
        self.since_said = 0 if (self.saying is not None or self.own) else getattr(self, "since_said", 10 ** 6) + 1
        if self.saying is not None and self.sound is not None:  # while a gesture sounds, what is heard is kept for its map
            self.said_codes.append(self.sound.x.copy())
            if not self.own:  # the gesture has ended: Hebb's rule, the running mean of what was heard while it sounded
                u0 = self.saying
                if self.amap is None:
                    self.amap = np.zeros((len(self.habit), len(self.sound.x)), np.float32)
                self.amap_n[u0] += 1
                self.amap[u0] += (np.mean(self.said_codes, 0) - self.amap[u0]) / self.amap_n[u0]
                self.saying, self.said_codes = None, []
        drive = self.habit / self.habit.max()
        strength = 1.0
        if self.echo_tau and self.sound is not None:
            self.echo_strength *= np.exp(-DT / self.echo_tau)
            loud = sound_row is not None and float(np.sum(sound_row)) > 1e-3
            if loud and self.saying is None:  # a stretch of sound goes on (not one's own: suppressed while saying)
                self.stretch = self.sound.x.copy() if self.stretch is None else self.stretch + self.sound.x
                self.stretch_n += 1
            elif self.stretch is not None and not loud:  # it ended: it becomes the echo, whole
                self.echo, self.echo_strength, self.stretch, self.stretch_n = self.stretch / self.stretch_n, 1.0, None, 0
        if self.imitate and self.amap is not None and self.sound is not None and self.saying is None:
            if self.echo_tau:
                src = self.stretch / self.stretch_n if self.stretch is not None else self.echo
                strength = 1.0 if self.stretch is not None else self.echo_strength
            else:
                src = self.sound.x
            src = src if src is not None else 0 * self.sound.x
            heard_now = src / (np.linalg.norm(src) + 1e-8)
            c = self.amap @ heard_now / (np.linalg.norm(self.amap, axis=1) + 1e-8)
            known = self.amap_n > 0
            m = np.maximum(c - c[known].mean(), 0) * known  # above the average gesture, as the filters answer above the average kind
            if m.max() > 0:
                drive = drive + strength * m / m.max()  # each source taken relative to its own strongest, then together
                world_loud = sound_row is not None and float(np.sum(sound_row)) > 1e-3
                if self.heard_habit and world_loud and self.since_said >= 10:
                    # covert imitation leaves habits, while sound comes in and not from one's own voice: the suppression
                    # lasts while the hearing population still holds it (100 ms, twice its time constant). Tried without
                    # both (2026-11-26): the habits fed on the mouth's own echo, even its faint tail.
                    self.habit += self.heard_habit * m / m.max()
        u = self.mouth.step(drive) if self.saying is None else None
        if u is not None:
            self.habit[u] += self.habit_step  # a gesture made becomes a little more likely (habit)
            events.append(("sound", u))
            if self.tract is not None:
                from .cochlea import Cochlea
                self._ear = getattr(self, "_ear", None) or Cochlea()
                self.own = list(self._ear.hear(self.tract.say(*self.gestures[u], ms=200)))
                self.saying, self.said_codes = u, []
        for e in events:
            self.log.append((round(self.t, 3),) + e)
        return events
