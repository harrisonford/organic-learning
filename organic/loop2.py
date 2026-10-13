"""Loop brain, revised after the fly connectome (docs/FLY.md).

Changes from loop.py, each tied to a finding:

  few broad feedback neurons -> three control ports driven by the orchestrator:
      where      a gaze-target map (move),
      how long   a "stay" command (keep fixating; with photoreceptor noise,
                 staying genuinely averages evidence),
      which      8 gain neurons, each reaching a broad fixed random third of the
                 eye's output channels and boosting them (centrifugal-like).
  reflex is for danger -> the reflex reacts to *change* in the coarse glance
      (onset, motion, looming), not to static contrast; in a still scene it is
      close to silent.
  shallow and wide -> the orchestrator reads the eye's parallel channels directly.
  commands are a library -> all eye commands (25 targets, stay, 8 gains, no
      gain) are output neurons of the orchestrator, credited by one rule.
  the eye is told about movement -> efference copy: the last command issued is
      part of the situation the orchestrator sees.

Per tick: the eye delivers coarse (tick 1 of a fixation), medium (tick 2) and
then fine samples (tick 3 and every tick it stays). After each fine sample the
orchestrator chooses: stay, or move (and which gain to apply at the next
fixation). At every tick it silently plans an answer and speaks when its
confidence passes its patience. The only teacher for commands is the outcome
(right/wrong, how long), through eligibility traces and per-command reward
prediction errors.
"""

import numpy as np

from .cortex import Cortex
from .ear import Ear
from .peye import RawSensor


class NoisyEye:
    """Fixed organ with control ports. Views are noisy (photoreceptor noise)."""

    def __init__(self, parts, size=40, noise=0.1, gains=8, seed=0):
        self.parts = parts
        self.sensor = RawSensor(size=size)
        self.T = len(self.sensor.targets)
        self.D = parts.out_dim
        self.noise = noise
        rng = np.random.default_rng(seed + 99)
        # each gain (centrifugal) neuron reaches a broad, fixed third of the channels
        self.gain_patterns = (rng.random((gains, self.D)) < 0.33).astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def view(self, img, target, level, gain=None):
        noisy = np.clip(img + self.rng.normal(0, self.noise, img.shape).astype(np.float32), 0, 1)
        code = self.parts.encode(self.sensor.views(noisy, target)[level])
        if gain is not None:
            code = code * (1.0 + self.gain_patterns[gain])
        n = np.linalg.norm(code)
        return code / n if n > 0 else code

    def glance(self, img):
        """Coarse luminance at each target region (for the change reflex)."""
        lum = np.clip(img + self.rng.normal(0, self.noise, img.shape).astype(np.float32), 0, 1).mean(axis=2)
        return np.array([self.sensor._crop(lum[..., None], cx, cy, 8).mean() for cx, cy in self.sensor.targets], np.float32)


class LoopBrain2:
    def __init__(self, parts, seed=0, goal_loop=True, reflex_loop=True, max_ticks=12, size=40, gains=8):
        self.eye = NoisyEye(parts, size=size, gains=gains, seed=seed)
        self.ear = Ear(seed=seed + 7)
        T, D = self.eye.T, self.eye.D
        self.commands = [f"eye:{t}" for t in range(T)] + ["eye:stay"] + [f"gain:{k}" for k in range(gains)] + ["gain:none"]
        self.dim = self.ear.width + 3 * D + T + len(self.commands)
        self.orch = Cortex(dim=self.dim, neurons_per_cluster=8, theta_familiar=0.95, associate=False, maturation=5.0, seed=seed)
        self.cmd = {c: self.orch.output_neuron(c) for c in self.commands}
        self.move_group = np.array([self.cmd[f"eye:{t}"] for t in range(T)] + [self.cmd["eye:stay"]])
        self.gain_group = np.array([self.cmd[f"gain:{k}"] for k in range(gains)] + [self.cmd["gain:none"]])
        self.value = {c: 0.0 for c in self.cmd.values()}
        self.goal_loop, self.reflex_loop = goal_loop, reflex_loop
        self.max_ticks = max_ticks
        self.gains = gains
        self.rng = np.random.default_rng(seed)
        self.age = 0

    def _situation(self, q_code, seen, target, last_cmd):
        g = np.zeros(self.eye.T, np.float32)
        g[target] = 1.0
        e = np.zeros(len(self.commands), np.float32)  # efference copy of the last command
        if last_cmd is not None:
            e[self.commands.index(last_cmd)] = 1.0
        n = np.linalg.norm(seen)
        s = np.concatenate([q_code, seen / n if n > 0 else seen, 0.5 * g, 0.5 * e])
        return s / np.linalg.norm(s)

    def _words(self):
        return np.array([i for i, o in enumerate(self.orch.outputs) if o.startswith("word:")], np.int64)

    def _choose(self, s, group, extra, explore):
        """The orchestrator's vote (goal loop) plus any organ-local drive (reflex)."""
        drive = np.array(extra, np.float32)
        r = self.orch.forward(s, group) if self.goal_loop else None
        if r is not None and r["probs"] is not None and r["raw"].sum() > 0:
            drive = drive + r["raw"] / (r["raw"].max() + 1e-8)
        if explore:
            p = np.exp((drive - drive.max()) / 0.15)
            k = int(self.rng.choice(len(drive), p=p / p.sum()))
        else:
            k = int(np.argmax(drive + 1e-6 * self.rng.standard_normal(len(drive))))
        return k, r

    def live(self, img, question, answer=None, learn=True, explore=True, time_cost=0.02):
        """Returns (answer said, tick, reward, gaze path, number of stays, gains used)."""
        img = np.asarray(img, np.float32)
        q_code = self.ear.question(self.ear.units(question, learn))
        if answer is not None:
            self.orch.output_neuron(f"word:{answer}")
        words = self._words()
        T, D = self.eye.T, self.eye.D
        seen = np.zeros(3 * D, np.float32)
        target, phase, gain, last_cmd = 12, 0, None, None
        fine_sum, fine_n = np.zeros(D, np.float32), 0
        prev_glance = None
        path, stays, gains_used, choices = [12], 0, [], []
        said, snapshot, tick = None, None, 0
        while tick < self.max_ticks:
            level = min(phase, 2)
            code = self.eye.view(img, target, level, gain)
            if level == 2:  # staying accumulates fine evidence
                fine_sum += code
                fine_n += 1
                code = fine_sum / fine_n
            seen = 0.7 * seen
            seen[level * D : (level + 1) * D] = code if level == 2 else seen[level * D : (level + 1) * D] + code
            tick += 1
            # silently plan an answer
            s = self._situation(q_code, seen, target, last_cmd)
            r = self.orch.forward(s, words) if len(words) else None
            if r is not None and r["probs"] is not None:
                total = r["g"].sum()
                patience = float((self.orch.patience[r["active"]] * r["g"]).sum() / total) if total else 1.0
                if r["probs"].max() >= patience or tick == self.max_ticks:
                    said = self.orch.outputs[words[int(np.argmax(r["probs"]))]].split(":", 1)[1]
                    snapshot = (s, r)
                    break
            if level < 2:
                phase += 1
                continue
            # after a fine sample: stay, or move (and set the gain for the next fixation)
            glance = self.eye.glance(img)
            reflex = np.zeros(T + 1, np.float32)
            if self.reflex_loop and prev_glance is not None:
                change = np.abs(glance - prev_glance)
                reflex[:T] = change / (change.max() + 1e-8) * min(1.0, change.max() * 10)
            prev_glance = glance
            reflex[target] -= 1.0  # "moving" to where it already is means nothing
            k, rm = self._choose(s, self.move_group, reflex, explore)
            if rm is not None and rm.get("a") is not None:
                choices.append((tick, rm["active"], rm["g"], rm["a"], int(self.move_group[k])))
            if k == T:  # stay
                stays += 1
                last_cmd = "eye:stay"
                continue
            target, phase = k, 0
            fine_sum, fine_n = np.zeros(D, np.float32), 0
            path.append(target)
            last_cmd = f"eye:{target}"
            kg, rg = self._choose(s, self.gain_group, np.zeros(self.gains + 1), explore)
            gain = None if kg == self.gains else kg
            gains_used.append(gain)
            if rg is not None and rg.get("a") is not None:
                choices.append((tick, rg["active"], rg["g"], rg["a"], int(self.gain_group[kg])))
        reward = None
        if answer is not None:
            correct = said == answer
            reward = (1.0 if correct else 0.0) - time_cost * tick
            if learn:
                self.age += 1
                s = snapshot[0] if snapshot else self._situation(q_code, seen, target, last_cmd)
                self.orch.learn(s, self.orch.out_index[f"word:{answer}"], words)  # imitation
                if snapshot:
                    self._tune_patience(snapshot[1], correct)
                self._credit(choices, reward, tick)
        return said, tick, reward, path, stays, gains_used

    def _credit(self, choices, reward, t_end, eta=0.1, fade=0.85):
        """Eligibility x reward prediction error for the command that was issued."""
        for t, active, g, a, col in choices:
            delta = reward - self.value[col]
            self.value[col] += 0.02 * delta
            e = fade ** (t_end - t)
            for c, gc, ac in zip(active, g, a):
                self.orch.Wout[c, :, col] = np.clip(self.orch.Wout[c, :, col] + eta * delta * e * gc * ac, 0.0, 1.0)

    def _tune_patience(self, r, correct, eta=0.05):
        step = -0.2 * eta if correct else +eta
        total = r["g"].sum() or 1.0
        for c, gc in zip(r["active"], r["g"]):
            self.orch.patience[c] = float(np.clip(self.orch.patience[c] + step * gc / total, 0.02, 0.99))
