"""Organs with control ports, an orchestrator that drives them, and two loops.

    question (raw bytes) ──► ear ───────────────┐
                                                ▼
    eye ──views──► sensory clump ──────► ORCHESTRATOR (clumps that grow by affinity)
     ▲                                     │            │            │
     │ control port "where"                │ goal loop  │ answer     │ gate (patience)
     └──── motor map ◄─────────────────────┘            ▼            ▼
              ▲                                     word-forms    speak / wait
              └── reflex loop: salience in the coarse view (organ-local, innate)

Organs are fixed. The eye exposes a control port: a map of possible gaze
targets. Two loops drive it and compete:
  - reflex (short loop, inside the organ): the coarse view's salience pulls
    the eye, fast and the same for every question (colliculus-like);
  - goal (long loop, through the orchestrator): clusters that stand for the
    current situation (question + what has been seen) vote for targets.

The only teacher for the goal loop is the outcome of the answer (right,
wrong, how long it took). Each gaze choice leaves an eligibility trace on the
synapses that voted for it; when the outcome arrives, those synapses change
by the reward prediction error *for that action* (what this look earned vs
what looking there usually earns), faded by how long ago it happened.

Answers are learned by imitation (the person's answer is heard afterwards),
and the patience gate decides when to speak, tuned by the same outcome.
"""

import numpy as np

from .area import SensoryArea
from .cortex import Cortex
from .ear import Ear
from .peye import RawSensor


class Eye:
    """Fixed organ: raw coarse/medium/fine views around a gaze target, parts-coded,
    a gaze port (target map) and an innate salience reflex on the coarse view."""

    def __init__(self, parts, size=40):
        self.parts = parts
        self.sensor = RawSensor(size=size)
        self.T = len(self.sensor.targets)
        self.D = parts.out_dim

    def views(self, img, target):
        return [self.parts.encode(v) for v in self.sensor.views(img, target)]

    def salience(self, img):
        """Reflex: how much the image differs from its surroundings at each target."""
        sal = np.zeros(self.T, np.float32)
        lum = img.mean(axis=2)
        for t, (cx, cy) in enumerate(self.sensor.targets):
            patch = self.sensor._crop(img, cx, cy, 8).mean(axis=2)
            sal[t] = abs(patch.mean() - lum.mean()) + patch.std()
        return sal / (sal.max() + 1e-8)


class LoopBrain:
    def __init__(self, parts, seed=0, goal_loop=True, reflex_loop=True, fixations=4, size=40):
        self.eye = Eye(parts, size=size)
        self.ear = Ear(seed=seed + 7)
        T, D = self.eye.T, self.eye.D
        self.sense = SensoryArea("seen", 3 * D + T, vigilance=0.8, seed=seed, axon_dim=256, population=16, temp=0.1, capacity=4096)
        self.dim = self.ear.width + self.sense.dim + T
        self.orch = Cortex(dim=self.dim, neurons_per_cluster=8, theta_familiar=0.95, associate=False, maturation=5.0, seed=seed)
        self.eye_out = [self.orch.output_neuron(f"eye:{t}") for t in range(T)]
        self.goal_loop, self.reflex_loop = goal_loop, reflex_loop
        self.fixations = fixations
        self.value = np.zeros(T, np.float32)  # what looking at each target usually earns
        self.rng = np.random.default_rng(seed)
        self.age = 0

    # ------------------------------------------------------------------ helpers

    def _seen(self, state):
        if not len(state) or state.sum() <= 0:
            return np.zeros(self.sense.dim, np.float32)
        v = state @ self.sense.Z[: len(state)]
        return v / (np.linalg.norm(v) + 1e-8)

    def _situation(self, q_code, state, target):
        g = np.zeros(self.eye.T, np.float32)
        g[target] = 1.0
        s = np.concatenate([q_code, self._seen(state), 0.5 * g])
        return s / np.linalg.norm(s)

    def _words(self):
        return np.array([i for i, o in enumerate(self.orch.outputs) if o.startswith("word:")], np.int64)

    # -------------------------------------------------------------------- live

    def live(self, img, question, answer=None, learn=True, explore=True, time_cost=0.02):
        """One moment: look, maybe answer at any tick, learn from the outcome.

        Returns (answer said, tick, reward, gaze path)."""
        img = np.asarray(img, np.float32)
        q_code = self.ear.question(self.ear.units(question, learn))
        if answer is not None:
            self.orch.output_neuron(f"word:{answer}")
        words = self._words()
        eye_group = np.array(self.eye_out)
        reflex = self.eye.salience(img) if self.reflex_loop else np.zeros(self.eye.T, np.float32)
        state = np.zeros(0, np.float32)
        latest = [np.zeros(self.eye.D, np.float32) for _ in range(3)]
        target, path, choices = 12, [12], []  # choices: (tick, clusters, weights, neuron acts, target)
        said, tick, snapshot = None, 0, None
        for f in range(self.fixations):
            if f > 0:  # choose where to look next: goal loop + reflex loop compete
                s = self._situation(q_code, state, target)
                goal = np.zeros(self.eye.T, np.float32)
                r = self.orch.forward(s, eye_group) if self.goal_loop else None
                if r is not None and r["probs"] is not None and r["raw"].sum() > 0:
                    goal = r["raw"] / (r["raw"].max() + 1e-8)
                drive = goal + 0.5 * reflex
                drive[target] -= 1.0  # the eye cannot "move" to where it already is
                if explore:
                    p = np.exp((drive - drive.max()) / 0.15)
                    target = int(self.rng.choice(self.eye.T, p=p / p.sum()))
                else:
                    target = int(np.argmax(drive))
                if r is not None and r.get("a") is not None:
                    choices.append((tick, r["active"], r["g"], r["a"], target))
                path.append(target)
            for level, code in enumerate(self.eye.views(img, target)):
                latest[level] = code
                state = np.concatenate([state, np.zeros(max(0, self.sense.C - len(state)), np.float32)]) * 0.7
                x = np.concatenate([latest[0], latest[1], latest[2], 0.5 * np.eye(self.eye.T, dtype=np.float32)[target]])
                idx, act = self.sense.perceive(x, learn)
                state = np.concatenate([state, np.zeros(max(0, self.sense.C - len(state)), np.float32)])
                state[idx] += act
                tick += 1
                # silently plan an answer; speak when sure enough
                s = self._situation(q_code, state, target)
                r = self.orch.forward(s, words) if len(words) else None
                if r is not None and r["probs"] is not None:
                    total = r["g"].sum()
                    patience = float((self.orch.patience[r["active"]] * r["g"]).sum() / total) if total else 1.0
                    last = f == self.fixations - 1 and level == 2
                    if r["probs"].max() >= patience or last:
                        said = self.orch.outputs[words[int(np.argmax(r["probs"]))]].split(":", 1)[1]
                        snapshot = (s, r)
                        break
            if said is not None:
                break
        reward = None
        if answer is not None:
            correct = said == answer
            reward = (1.0 if correct else 0.0) - time_cost * tick
            if learn:
                self.age += 1
                s = snapshot[0] if snapshot else self._situation(q_code, state, target)
                self.orch.learn(s, self.orch.out_index[f"word:{answer}"], words)  # imitation
                if snapshot:
                    self._tune_patience(snapshot[1], correct, tick)
                self._credit_gaze(choices, reward, tick)
        return said, tick, reward, path

    # ---------------------------------------------------------------- learning

    def _credit_gaze(self, choices, reward, t_end, eta=0.1, fade=0.85):
        """Three-factor rule: eligibility (which synapses voted for the chosen
        target, fading with time) x reward prediction error for that target."""
        for t, active, g, a, target in choices:
            delta = reward - self.value[target]
            self.value[target] += 0.02 * delta
            e = fade ** (t_end - t)
            col = self.eye_out[target]
            for c, gc, ac in zip(active, g, a):
                w = self.orch.Wout[c, :, col] + eta * delta * e * gc * ac
                self.orch.Wout[c, :, col] = np.clip(w, 0.0, 1.0)

    def _tune_patience(self, r, correct, tick, eta=0.05):
        step = -0.2 * eta if correct else +eta
        total = r["g"].sum() or 1.0
        for c, gc in zip(r["active"], r["g"]):
            self.orch.patience[c] = float(np.clip(self.orch.patience[c] + step * gc / total, 0.02, 0.99))
