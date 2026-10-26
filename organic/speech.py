"""Speaking: a speech motor area built by babbling (design: docs/SPEECH.md).

The loop brain (loop2) answers by choosing one output neuron per whole answer.
This brain speaks: it produces motor units (byte gestures) one per tick after
it decides to speak, hears itself, and stops when it chooses silence.

  Mouth            motor units: one per byte at birth; chunks of units said
                   together often become one unit (motor chunking, like the
                   ear's units on the hearing side). Articulation is noisy at
                   first (a gesture sometimes comes out as another sound) and
                   gets precise with practice.
  auditory-motor   for every motor unit, what it sounds like to the ear: a
  map              running average of the ear's code of what actually came out
                   each time it was said (learned from babbling and from all
                   own speech, never wired).
  babbling         spontaneous strings of 1-4 motor units, biased toward units
                   that sound like recently heard speech (drift toward the
                   language heard).
  imitation        the adult's answer, as the child can say it with the units
                   it has (longest pieces first), followed by silence: at each
                   speaking tick, the target is the next unit.

Everything else (eye, cortex, orchestrator rules, patience, eye commands,
sleep, budgets) is the loop brain's, unchanged.

  copy route       (milestone 3) the heard utterance stays as an auditory
                   trace; while speaking, each motor unit is primed by how well
                   its learned sound matches what comes next in the trace,
                   starting at any heard unit and then following on from where
                   the last unit matched (self-hearing moves the place along).
                   It works only as well as the auditory-motor map: without
                   babbling, nothing to copy with.

The live() loop is a copy of LoopBrain2.live with the answer part replaced, so
that lives running with LoopBrain2 are not affected.
"""

import numpy as np

from .loop2 import LoopBrain2

SILENCE = b""


class Mouth:
    def __init__(self, width, chunk_after=30, max_units=4000, seed=0):
        self.units = [bytes([b]) for b in range(32, 127)] + [b"\n", SILENCE]
        self.index = {u: i for i, u in enumerate(self.units)}
        self.amap = np.zeros((len(self.units), width), np.float32)
        self.amap_n = np.zeros(len(self.units), np.int64)
        self.practice = np.zeros(len(self.units), np.int64)
        self.pairs = {}
        self.chunk_after, self.max_units = chunk_after, max_units
        self.rng = np.random.default_rng(seed)

    def add(self, unit):
        if unit in self.index:
            return False
        self.index[unit] = len(self.units)
        self.units.append(unit)
        self.amap = np.vstack([self.amap, np.zeros((1, self.amap.shape[1]), np.float32)])
        self.amap_n = np.append(self.amap_n, 0)
        self.practice = np.append(self.practice, 0)
        return True

    def articulate(self, unit):
        """What actually comes out: early gestures sometimes slip into another sound."""
        if unit == SILENCE:
            return SILENCE
        i = self.index[unit]
        slip = 0.3 / (1.0 + self.practice[i] / 50.0)
        self.practice[i] += 1
        out = bytearray(unit)
        for j in range(len(out)):
            if self.rng.random() < slip:
                out[j] = int(self.rng.integers(97, 123))  # a nearby, vowel/consonant-like slip
        return bytes(out)

    def hear_self(self, unit, came_out, ear):
        """Auditory-motor map: the sound of this gesture is what the ear heard when it was made."""
        if unit == SILENCE:
            return
        i = self.index[unit]
        self.amap_n[i] += 1
        self.amap[i] += (ear.sound(came_out) - self.amap[i]) / min(self.amap_n[i], 200)

    def decompose(self, text):
        """Imitation with the gestures one has: longest known pieces first."""
        data, out, k = text.encode("utf-8"), [], 0
        longest = max(len(u) for u in self.units)
        while k < len(data):
            for n in range(min(longest, len(data) - k), 0, -1):
                if data[k : k + n] in self.index:
                    out.append(data[k : k + n])
                    k += n
                    break
            else:  # a byte the mouth cannot make: skipped
                k += 1
        return out

    def rehearse(self, units):
        """Units said one after the other often fuse into one gesture (motor chunking)."""
        new = []
        for a, b in zip(units, units[1:]):
            if a == SILENCE or b == SILENCE:
                continue
            n = self.pairs[(a, b)] = self.pairs.get((a, b), 0) + 1
            if n == self.chunk_after and len(self.units) < self.max_units and self.add(a + b):
                # a new chunk is as skilled as its weaker part, and sounds like its parts in a row
                i = self.index[a + b]
                self.practice[i] = min(self.practice[self.index[a]], self.practice[self.index[b]])
                self.amap[i] = self.amap[self.index[a]] + self.amap[self.index[b]]
                self.amap[i] /= np.linalg.norm(self.amap[i]) + 1e-8
                new.append(a + b)
        return new


class SpeakingBrain(LoopBrain2):
    max_utterance = 16

    def __init__(self, parts, **kw):
        super().__init__(parts, **kw)
        self.mouth = Mouth(self.ear.width, seed=kw.get("seed", 0) + 5)
        # the situation also holds what I have said so far (efference copy + self-hearing)
        self.dim += self.ear.width
        self.orch.dim = self.dim
        for arr in ("Z", "Win"):  # no clusters exist yet; widen their storage
            a = getattr(self.orch, arr)
            setattr(self.orch, arr, np.zeros(a.shape[:-1] + (self.dim,), np.float32))
        self.say = {u: self.orch.output_neuron(self._name(u)) for u in self.mouth.units}
        # whether to let the copy route in is a choice, like the eye's gain neurons, credited by outcome
        self.copy_group = np.array([self.orch.output_neuron("speak:copy"), self.orch.output_neuron("speak:own")])
        for c in self.copy_group:
            self.value[int(c)] = 0.0
        self.heard_recent = np.zeros(self.ear.width, np.float32)
        self.babbles = 0

    @staticmethod
    def _name(u):
        return "say:" + u.hex()

    def _motor_group(self):
        return np.array([self.say[u] for u in self.mouth.units], np.int64)

    def _speech_situation(self, base, units):
        said = self.ear.said(list(units)) if units else np.zeros(self.ear.width, np.float32)
        s = np.concatenate([base, said])
        return self._adapt(s / np.linalg.norm(s), self.learning)

    def _base(self, q_code, seen, target, last_cmd, cortex):
        g = np.zeros(self.eye.T, np.float32)
        g[target] = 1.0
        e = np.zeros(len(self.commands), np.float32)
        if last_cmd is not None:
            e[self.commands.index(last_cmd)] = 1.0
        n = np.linalg.norm(seen)
        parts = [q_code, seen / n if n > 0 else seen, 0.5 * g, 0.5 * e]
        if self.cortex_dims:
            parts.append(cortex if cortex is not None else np.zeros(sum(self.cortex_dims), np.float32))
        return np.concatenate(parts)

    def _situation(self, q_code, seen, target, last_cmd, cortex=None):  # used by the eye's command choices
        return self._speech_situation(self._base(q_code, seen, target, last_cmd, cortex), ())

    # ----------------------------------------------------------- babbling

    def babble(self, n=1, explore=True):
        """Spontaneous sounds, heard by the ear; builds the auditory-motor map."""
        units = self.mouth.units
        sims = self.mouth.amap @ self.heard_recent
        for _ in range(n):
            p = np.exp(3.0 * sims) if explore else np.ones(len(units))
            p[self.mouth.index[SILENCE]] = 0
            p = p / p.sum()
            seq = [units[i] for i in self.rng.choice(len(units), int(self.rng.integers(1, 5)), p=p)]
            for u in seq:
                self.mouth.hear_self(u, self.mouth.articulate(u), self.ear)
            self.babbles += 1
        return b"".join(seq).decode("utf-8", "replace")

    copy_gain = 1.0  # strength of the hearing -> saying route (0: no copy route)

    def _segment_sound(self, seg):
        """The ear's code of a stretch of the heard trace (not kept as a unit)."""
        new = seg not in self.ear.codes
        v = self.ear.sound(seg)
        if new:
            del self.ear.codes[seg]
        return v

    def _prime(self, trace, positions):
        """Copy route: for each motor unit, how well its learned sound (auditory-motor
        map) matches what comes next in the heard trace at any of `positions`.
        Returns priming per unit (group order) and the position each unit matched at."""
        units = self.mouth.units
        amap = self.mouth.amap / (np.linalg.norm(self.mouth.amap, axis=1, keepdims=True) + 1e-8)
        prime = np.zeros(len(units), np.float32)
        where = np.full(len(units), -1, np.int64)
        lengths = {}
        for i, u in enumerate(units):
            if u:
                lengths.setdefault(len(u), []).append(i)
        for p in positions:
            for L, idx in lengths.items():
                if p + L > len(trace):
                    continue
                m = amap[idx] @ self._segment_sound(trace[p : p + L])
                better = m > prime[idx]
                prime[np.array(idx)[better]] = m[better]
                where[np.array(idx)[better]] = p
        return np.maximum(prime, 0) ** 2, where

    def _speak(self, base, r0, explore, trace=b"", starts=()):
        """Say one unit per tick from the first choice until silence. The orchestrator's
        vote is joined by the copy route's priming from the heard trace."""
        group = self._motor_group()
        units, came = [], []
        r = r0
        sil = self.mouth.index[SILENCE]
        pos, unit_end = None, None
        ends = sorted(set(starts[1:]) | {len(trace)})
        for _ in range(self.max_utterance):
            if r is None or r["probs"] is None:
                break
            drive = r["probs"].copy()
            if self.copy_gain and trace:
                prime, where = self._prime(trace, starts if pos is None else [pos])
                if pos is not None and pos >= unit_end:
                    prime[sil] = 1.0  # the heard unit being repeated is over
                drive = drive + self.copy_gain * prime
            if explore:
                p = np.exp(np.log(drive + 1e-9) / 0.5)
                k = int(self.rng.choice(len(group), p=p / p.sum()))
            else:
                k = int(np.argmax(drive))
            u = self.mouth.units[k]
            if u == SILENCE:
                break
            if self.copy_gain and trace and where[k] >= 0:  # follow the heard trace from where this matched
                pos = int(where[k]) + len(u)
                unit_end = next((e for e in ends if e > where[k]), len(trace))
            out = self.mouth.articulate(u)
            self.mouth.hear_self(u, out, self.ear)
            units.append(u)
            came.append(out)
            r = self.orch.forward(self._speech_situation(base, units), group)
        return units, b"".join(came).decode("utf-8", "replace").strip()

    # --------------------------------------------------------------- live

    def live(self, img, question, answer=None, learn=True, explore=True, time_cost=0.02, deadline=None, grade=None):
        """As LoopBrain2.live, but the answer is spoken unit by unit. Returns the same tuple."""
        img = np.asarray(img, np.float32)
        self.learning = learn
        heard = self.ear.units(question, learn)
        q_code = self.ear.question(heard)
        self.heard_recent = 0.9 * self.heard_recent + 0.1 * q_code
        group = self._motor_group()
        T, D = self.eye.T, self.eye.D
        seen = np.zeros(3 * D, np.float32)
        target, phase, gain, last_cmd = 12, 0, None, None
        fine_sum, fine_n = np.zeros(D, np.float32), 0
        prev_glance = None
        path, stays, gains_used, choices = [12], 0, [], []
        said, snapshot, tick, spoken = None, None, 0, []
        codes = [c for c in self.cortex.code(img, grid=1)[1:]] if self.cortex is not None else None
        while tick < self.max_ticks:
            level = min(phase, 2)
            code = self.eye.view(img, target, level, gain)
            if level == 2:
                fine_sum += code
                fine_n += 1
                code = fine_sum / fine_n
            seen = 0.7 * seen
            seen[level * D : (level + 1) * D] = code if level == 2 else seen[level * D : (level + 1) * D] + code
            tick += 1
            base = self._base(q_code, seen, target, last_cmd, self._cortex_now(codes, tick))
            s = self._speech_situation(base, ())
            r = self.orch.forward(s, group)
            if r is not None and r["probs"] is not None:
                total = r["g"].sum()
                patience = float((self.orch.patience[r["active"]] * r["g"]).sum() / total) if total else 1.0
                if r["probs"].max() >= patience or tick == self.max_ticks:
                    snapshot = (s, r, base)
                    starts = list(np.cumsum([0] + [len(h) for h in heard[:-1]]))
                    kc, rc = self._choose(s, self.copy_group, np.zeros(2), explore)
                    if rc is not None and rc.get("a") is not None:
                        choices.append((tick, rc["active"], rc["g"], rc["a"], int(self.copy_group[kc])))
                    self.copied = kc == 0
                    spoken, said = self._speak(base, r, explore and learn, trace=question.encode("utf-8") if kc == 0 else b"", starts=starts)
                    tick += len(spoken)  # speaking takes time
                    break
            if level < 2:
                phase += 1
                continue
            glance = self.eye.glance(img)
            reflex = np.zeros(T + 1, np.float32)
            if self.reflex_loop and self.reflex == "contrast":
                reflex[:T] = 0.5 * self.eye.salience(img)
            elif self.reflex_loop and prev_glance is not None:
                change = np.abs(glance - prev_glance)
                reflex[:T] = change / (change.max() + 1e-8) * min(1.0, change.max() * 10)
            prev_glance = glance
            reflex[target] -= 1.0
            if not self.use_stay:
                reflex[T] = -1e9
            k, rm = self._choose(s, self.move_group, reflex, explore)
            if rm is not None and rm.get("a") is not None:
                choices.append((tick, rm["active"], rm["g"], rm["a"], int(self.move_group[k])))
            if k == T:
                stays += 1
                last_cmd = "eye:stay"
                continue
            target, phase = k, 0
            fine_sum, fine_n = np.zeros(D, np.float32), 0
            path.append(target)
            last_cmd = f"eye:{target}"
            if self.use_gains:
                kg, rg = self._choose(s, self.gain_group, np.zeros(self.gains + 1), explore)
                gain = None if kg == self.gains else kg
                gains_used.append(gain)
                if rg is not None and rg.get("a") is not None:
                    choices.append((tick, rg["active"], rg["g"], rg["a"], int(self.gain_group[kg])))
        reward = None
        if answer is not None:
            value = float(grade(said)) if grade is not None else (1.0 if said == answer else 0.0)
            correct = value >= 0.999
            if deadline is not None and tick > deadline:
                value *= 0.5 ** (tick - deadline)
            reward = value - time_cost * tick
            if learn:
                self.age += 1
                base = snapshot[2] if snapshot else self._base(q_code, seen, target, last_cmd, self._cortex_now(codes, tick))
                targets = self.mouth.decompose(answer)
                for u in self.mouth.rehearse(targets):  # new motor chunks get their output neurons
                    self.say[u] = self.orch.output_neuron(self._name(u))
                group = self._motor_group()
                for i, u in enumerate(targets + [SILENCE]):  # imitation, one unit per step
                    self.orch.learn(self._speech_situation(base, targets[:i]), self.say[u], group)
                if snapshot:
                    late = correct and reward < 1.0 - time_cost * tick - 1e-6
                    self._tune_patience(snapshot[1], correct, late)
                self._credit(choices, reward, tick)
                if self.sleep_cleanup and self.age % 2000 == 0:
                    merged, retired = self.orch.consolidate()
                    self.sleep_log = getattr(self, "sleep_log", []) + [(self.age, merged, retired, int(self.orch.C))]
                self.orch.maybe_prune()
        return said, tick, reward, path, stays, gains_used
