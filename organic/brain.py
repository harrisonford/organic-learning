"""A brain that looks at a scene, hears a question and answers in words.

    eye ─► form  ──► [shape area]  ──┐
        ─► color ──► [color area]  ──┤  cross-modal bindings (Hebbian)
        ─► where ──► [where area]  ──┤  -> priming of word neurons, through
        ─► count ──► [number area] ──┘     myelinated pathways
                                       │
    ear ─► question echo ──┐           ▼
        ─► own-speech echo ┼──► [speech area] ──► word neurons ──► mouth
    coincidence neurons ───┘     (grows by affinity)  (fatigue, category spread)

Nothing in here computes a gradient. What each part does:

- Sensory areas organize themselves without labels (novelty-driven growth).
- Words heard while an area is active leave Hebbian traces on its assemblies.
  Word neurons are homeostatic, so the area sends *lift* (how much more likely
  than usual a word is, given what it sees). Lifts from different areas
  multiply, like independent dendritic branches gating the same neuron.
- Coincidence neurons fire when a word just heard in the question is also one
  that vision is priming (NMDA-like AND), which is how "is it red?" can be
  answered yes or no.
- The speech area learns, by imitation of the answers it hears, *what kind of*
  word comes next given the question and what it has said so far. It grows a
  new cluster when nothing has affinity for the situation, and only clusters
  with affinity learn.
- Word neurons that compete for the same slot become associated, so word
  categories (colors, shapes, places) emerge from distribution alone, and
  activation spreads within a category. Priming then picks the member that
  matches what is seen, even in combinations never heard before.
- Word neurons fatigue after firing, which keeps speech from looping.
- Attention: in a scene with several objects the eye attends one at a time
  and moves on (inhibition of return) once that object's name has been said.
- A hippocampus stores episodes and replays them during sleep; sleep also
  merges redundant clusters and lets useless ones die.
"""

import numpy as np

from .area import SensoryArea
from .cortex import Cortex
from .ear import SILENCE, Ear
from .eye import Eye
from .hippocampus import Hippocampus
from .pathway import Pathway
from .sprout import SproutedArea

END = SILENCE  # an answer ends when the speaker falls silent


def norm_text(text):
    return " ".join(text.replace(",", " , ").replace("?", " ? ").split())


class Brain:
    hear_categories = False  # defaults for brains saved before these existed
    fatigue_decay = 0.4
    sprouting = False

    def __init__(self, seed=0, lam=1.5, spread=0.3, fatigue=0.9, coincidence_weight=0.8, category_blend=0.5, rem_weight=0.6, hear_categories=False, fatigue_decay=0.4, sprouting=False, stable_sprouts=False):
        self.eye = Eye()
        self.ear = Ear(seed=seed + 7)
        self.areas = {
            "shape": SensoryArea("shape", Eye.form_dim, vigilance=0.86),
            "color": SensoryArea("color", Eye.color_dim, vigilance=0.9),
            "where": SensoryArea("where", Eye.where_dim, vigilance=0.88),
            "number": SensoryArea("number", Eye.number_dim, vigilance=0.9),
            # fast, coarse (magnocellular) glance at the whole scene
            "gist_color": SensoryArea("gist_color", Eye.gist_color_dim, vigilance=0.9),
            "gist_where": SensoryArea("gist_where", Eye.gist_where_dim, vigilance=0.9),
            "gist_form": SensoryArea("gist_form", Eye.gist_form_dim, vigilance=0.9),
        }
        self.scene_areas = ("number", "gist_color", "gist_where", "gist_form")
        self.object_areas = ("shape", "color", "where")
        self.pathways = {k: Pathway(f"{k}->speech") for k in self.areas}
        self.co_dim = 12  # match / mismatch neurons (6 each)
        self.rem_dim = 6  # "how many salient things have I not attended yet"
        self.speech = Cortex(
            dim=2 * self.ear.width + self.co_dim + self.rem_dim,
            neurons_per_cluster=8,
            theta_familiar=0.97,
            competition_temp=0.02,
            associate=False,
            maturation=5.0,
            seed=seed,
        )
        self.hippocampus = Hippocampus(seed=seed + 11)
        self.lam = lam  # how strongly priming gates word neurons
        self.spread = spread  # activation spread within emergent word categories
        self.fatigue_strength = fatigue
        self.fatigue_decay = fatigue_decay  # how much of a word-form's fatigue remains after each unit
        self.co_weight = coincidence_weight
        self.category_blend = category_blend
        self.rem_weight = rem_weight
        # heard words also carry a sense of their emergent category, so
        # "either red or blue" and "either white or green" feel alike
        self.hear_categories = hear_categories
        self.vcap = 64
        self.baseline = np.zeros(self.vcap, np.float32)  # homeostatic average activity
        self.assoc = np.zeros((self.vcap, self.vcap), np.float32)  # word-word slot associations
        self.visual_episodes = 0
        self.age = 0
        # structural plasticity at the level of areas (see sprout.py)
        self.sprouting = sprouting
        self.sprouted = []  # list of (SproutedArea, Pathway)
        self.sprout_weight = 0.8
        self.stable_sprouts = stable_sprouts
        self.surprise_trace = 0.5
        self.now = None  # current tick within a moment (None = no time limit)
        self.time_cost = 0.02  # what each tick of waiting costs (felt via reward)
        self.surprise_window, self.surprise_history = [], []
        self.last_sprout = 0
        self.events = []

    def __setstate__(self, state):
        """Load brains saved by older versions: fill in parts they did not have yet."""
        self.__dict__.update(state)
        defaults = {"now": None, "time_cost": 0.02, "stable_sprouts": False, "surprise_trace": 0.5, "sprouted": [], "sprout_weight": 0.8, "surprise_window": [], "surprise_history": [], "last_sprout": 0, "events": []}
        for k, v in defaults.items():
            self.__dict__.setdefault(k, v)

    # -------------------------------------------------------------- lexicon

    def unit(self, u):
        """The word-form neuron for a unit the ear produced (grown if new)."""
        i = self.speech.output_neuron(u.hex())
        if i >= self.vcap:
            old = self.vcap
            while self.vcap <= i:
                self.vcap *= 2
            b = np.zeros(self.vcap, np.float32)
            b[:old] = self.baseline
            self.baseline = b
            a = np.zeros((self.vcap, self.vcap), np.float32)
            a[:old, :old] = self.assoc
            self.assoc = a
        return i

    @property
    def V(self):
        return len(self.speech.outputs)

    def lexicon(self):
        return np.arange(self.V)

    def name(self, i):
        return bytes.fromhex(self.speech.outputs[i])

    def known(self, u):
        return self.speech.out_index.get(u.hex())

    @staticmethod
    def text(units):
        return b"".join(units).decode("utf-8", "replace")

    # ----------------------------------------------------------- perception

    def look(self, image, learn):
        """Saccade over the scene and let the sensory areas respond."""
        if image is None:
            return {"fixations": [], "objects": [], "number": None, "gist": {}}
        fix = self.eye.look(image)
        objects = []
        for f in fix:
            objects.append(
                {
                    "shape": self.areas["shape"].perceive(f.form, learn),
                    "color": self.areas["color"].perceive(f.color, learn),
                    "where": self.areas["where"].perceive(np.concatenate([f.where]), learn),
                }
            )
        number = self.areas["number"].perceive(self.eye.number(len(fix)), learn) if fix else None
        glance = self.eye.glance(image)
        gist = {a: self.areas[a].perceive(glance[a], learn) for a in ("gist_color", "gist_where", "gist_form")}
        return {"fixations": fix, "objects": objects, "number": number, "gist": gist}

    @staticmethod
    def arrival(scene, area, k=0):
        """Tick at which a signal reaches cortex. The glance is fast; each
        saccade takes a tick; fine form needs the fovea to settle one more."""
        if area.startswith("gist"):
            return 1
        if area == "number":
            return 1 + len(scene["objects"])
        return (3 if area == "shape" else 2) + k

    def lifts(self, scene, attended):
        """Per-area lift over the lexicon, as delivered by each pathway."""
        V = self.V
        k = 0.02
        out = {}
        sources = {}
        now = self.now if self.now is not None else 10**9
        if scene["objects"] and attended is not None:
            for a in self.object_areas:
                if self.arrival(scene, a, attended) <= now:
                    sources[a] = scene["objects"][attended][a]
        if scene["number"] is not None and self.arrival(scene, "number") <= now:
            sources["number"] = scene["number"]
        for a, v in scene.get("gist", {}).items():
            if self.arrival(scene, a) <= now:
                sources[a] = v
        for a, (idx, act) in sources.items():
            e = self.areas[a].expectation(idx, act, V)
            if e is None:
                continue
            lift = (e + k) / (self.baseline[:V] + k)
            out[a] = self.resonate(np.clip(lift, 0.05, 10.0))
        return out

    def overlap(self):
        """How much each pair of word-forms shares auditory neurons (cached)."""
        V = self.V
        if getattr(self, "_overlap", None) is None or self._overlap.shape[0] != V:
            S = np.stack([self.ear.sound(self.name(i)) for i in range(V)])
            W = np.clip(S @ S.T, 0, 1) ** 2
            W[W < 0.15] = 0
            np.fill_diagonal(W, 1.0)
            self._overlap = W.astype(np.float32)
        return self._overlap

    def resonate(self, lift):
        """A word-form is primed through the auditory neurons it shares with
        primed forms: "red?" feels primed when "red" is (strongest input wins)."""
        return (self.overlap() * lift[None, :]).max(axis=1)

    def priming(self, lifts):
        """Facilitation of word neurons by each area, scaled by how much of the
        signal arrives in time. Priming only excites: an area that has never
        seen a white circle does not veto "circle"; suppression of the other
        candidates comes from competition between word neurons."""
        g = np.ones(self.V, np.float32)
        for a, lift in lifts.items():
            g *= np.maximum(lift, 1.0) ** (self.lam * self.pathways[a].gain)
        return g

    def comparator(self, q_units, lifts):
        """Match and mismatch neurons (prediction-error-like).

        A heard unit is *grounded* if some sensory area has a specific trace
        for it (the area knows what it looks like). Match neurons count heard,
        grounded units that what is seen right now supports; mismatch neurons
        count heard, grounded units that are expected but absent. Neither
        knows anything about "and", "or" or "not".
        """
        v = np.zeros(self.co_dim, np.float32)
        if not lifts:
            return v
        V = self.V
        k = 0.02
        support = np.ones(V, np.float32)
        grounded = np.zeros(V, bool)
        for a, lift in lifts.items():
            support *= np.maximum(lift, 1.0) ** self.pathways[a].gain
            area = self.areas[a]
            area._ensure(V=V)
            best = self.resonate(((area.Wname[: area.C, :V] + k) / (self.baseline[:V] + k)).max(axis=0))
            grounded |= best > 3.0
        idx = {self.known(u) for u in q_units} - {None}
        match = sum(1 for i in idx if grounded[i] and support[i] > 2.7)
        miss = sum(1 for i in idx if grounded[i] and support[i] <= 2.7)
        half = self.co_dim // 2
        centers = np.arange(half)
        v[:half] = np.exp(-0.5 * ((match - centers) / 0.4) ** 2)
        v[half:] = np.exp(-0.5 * ((miss - centers) / 0.4) ** 2)
        return v

    def inner_voice(self, w):
        """How my own word feels: its sound blended with the sounds of the words
        it substitutes for, so a word carries a sense of its emergent category."""
        v = self.ear.sound(w)
        i = self.known(w)
        if i is None or i >= self.V:
            return v
        row = self.assoc[i, : self.V]
        total = row.sum()
        if total <= 0:
            return v
        others = sum(row[j] * self.ear.sound(self.name(j)) for j in np.nonzero(row > 0.05)[0])
        blend = v + self.category_blend * others / max(total, 1.0)
        return blend / np.linalg.norm(blend) * np.linalg.norm(v)

    def speech_input(self, q_code, said, co, remaining, sprouted=()):
        parts = [q_code, self.ear.said(said, self.inner_voice), self.co_weight * co, self.rem_weight * remaining]
        for (area, path), code in zip(self.sprouted, sprouted):
            n = np.linalg.norm(code)  # each sprouted area's output, by pathway gain
            parts.append(self.sprout_weight * path.gain * (code / n if n > 0 else code))
        s = np.concatenate(parts)
        return s / np.linalg.norm(s)

    def _sprout_codes(self, scene, attention, q_code, co, learn):
        if not self.sprouted:
            return ()
        zero = np.zeros(64, np.float32)
        outs = {}
        if scene["objects"] and attention["at"] is not None:
            for a in self.object_areas:
                outs[a] = self.areas[a].output(*scene["objects"][attention["at"]][a])
        if scene["number"] is not None:
            outs["number"] = self.areas["number"].output(*scene["number"])
        sources = [q_code, co] + [outs.get(a, zero) for a in ("shape", "color", "where", "number")]
        return [area.respond(sources, learn) for area, _ in self.sprouted]

    def _monitor(self, surprise, window=500):
        """Chronic surprise that neurogenesis is not fixing -> grow a new area."""
        self.surprise_window.append(surprise)
        if len(self.surprise_window) < window:
            return
        now = float(np.mean(self.surprise_window))
        self.surprise_window = []
        prev = self.surprise_history[-1] if self.surprise_history else None
        self.surprise_history.append(now)
        stuck = prev is not None and (prev - now) < 0.03 * prev
        if self.sprouting and stuck and now > 0.15 and self.age - self.last_sprout >= 2000 and len(self.sprouted) < 3:
            self._sprout()

    def _sprout(self):
        name = f"sprout{len(self.sprouted) + 1}"
        blocks = [("question", self.ear.width), ("comparator", self.co_dim)] + [(a, 64) for a in ("shape", "color", "where", "number")]
        area = SproutedArea(name, blocks, seed=len(self.sprouted) + 101, stable=self.stable_sprouts)
        # a young area starts half-connected (critical period); whether its
        # pathway matures is then decided by the dopamine-like signal below
        self.sprouted.append((area, Pathway(f"{name}->speech", myelin=0.3 if self.stable_sprouts else 0.05)))
        self.speech.grow_inputs(area.out_dim)
        self.last_sprout = self.age
        self.events.append((self.age, f"sprouted {name}"))

    def word_gain(self, prime, fatigue):
        return prime * (1.0 - self.fatigue_strength * fatigue)

    # --------------------------------------------------------- one episode

    def _names(self, lifts):
        """The words that most specifically name the attended object."""
        return [int(np.argmax(lifts[a])) for a in ("shape", "color") if a in lifts]

    def _remaining_code(self, scene, attention):
        v = np.zeros(self.rem_dim, np.float32)
        if scene["objects"]:
            left = len(scene["objects"]) - len(attention["visited"])  # not yet named
            v[:] = np.exp(-0.5 * ((left - np.arange(self.rem_dim)) / 0.4) ** 2)
        return v

    def _shift(self, scene, attention, to=None):
        """Inhibition of return: leave the current object and fixate the next one."""
        if attention["at"] is not None:
            attention["visited"].add(attention["at"])
        if to is None:
            free = [k for k in range(len(scene["objects"])) if k not in attention["visited"]]
            to = free[0] if free else None  # saccades go to the most salient free object
        attention["at"] = to

    def _search(self, scene, q_words):
        """Feature-based attention: heard words that vision knows pull the eye
        to the object that matches them best; otherwise the most salient one."""
        if not scene["objects"]:
            return None
        if len(scene["objects"]) == 1:
            return 0
        heard = [i for i in (self.known(u) for u in q_words) if i is not None]
        if not heard:
            return 0
        # biased competition (Desimone & Duncan 1995): objects compete for
        # attention; heard words add a bias to the objects they describe
        scores = []
        for k in range(len(scene["objects"])):
            ls = self.lifts(scene, k)
            scores.append(sum(np.log(max(float(ls[a][heard].max()), 1.0)) for a in ("shape", "color", "where") if a in ls))
        order = np.argsort(scores)[::-1]
        if scores[order[0]] - scores[order[1]] > np.log(1.2):
            return int(order[0])
        return 0  # no clear bias: the most salient object wins

    def _joint_attention(self, scene, attention, w):
        """While listening, look at the object that the heard word is about."""
        wi = self.known(w)
        if len(scene["objects"]) < 2 or wi is None:
            return False
        best, best_lift = None, 2.0
        for k in range(len(scene["objects"])):
            if k in attention["visited"]:
                continue
            ls = self.lifts(scene, k)
            lift = max((ls[a][wi] for a in ("shape", "color") if a in ls), default=0.0)
            if lift > best_lift:
                best, best_lift = k, lift
        if best is not None and best != attention["at"]:
            self._shift(scene, attention, to=best)
            return True
        return False

    def live(self, image, question, answer=None, learn=True, temperature=0.0, max_words=16, rng=None, trace=None, tick=None, scene=None):
        """Experience a moment. With an answer: imitate it. Without: respond.

        trace: optional list; each speaking step appends its top candidates.
        tick: only signals that have arrived by this tick are available
        (None = everything). scene: an already perceived scene.
        """
        if learn:
            self.age += 1  # only lived experience counts, not tests
        self.now = tick
        if scene is None:
            scene = self.look(image, learn)
        q_words = self.ear.units(question, learn)
        for w in q_words:
            self.unit(w)
        target = None
        if answer is not None:
            target = self.ear.units(answer, learn) + [END]
            for w in target:
                self.unit(w)
        q_code = self.ear.question(q_words, self.inner_voice if self.hear_categories else None)
        group = self.lexicon()
        fatigue = np.zeros(self.vcap, np.float32)
        for w in q_words:  # just-heard words are a little fatigued too
            fatigue[self.known(w)] = 0.3
        attention = {"at": self._search(scene, q_words), "visited": set()}
        said, heard_by_object, surprises = [], {}, []
        confidence, involved = 1.0, {}
        correct = 0
        steps = len(target) if target else max_words
        for t in range(steps):
            lifts = self.lifts(scene, attention["at"])
            prime = self.priming(lifts)
            co = self.comparator(q_words, lifts)
            spr = self._sprout_codes(scene, attention, q_code, co, learn and target is not None)
            s = self.speech_input(q_code, said, co, self._remaining_code(scene, attention), spr)
            gain = self.word_gain(prime, fatigue[: self.V])
            lateral = self._lateral()
            if target:
                w = target[t]
                wi = self.known(w)
                r = self.speech.learn(s, wi, group, gain, lateral)
                correct += r["correct"]
                f = r["forward"]
                surprises.append(1.0 - float(f["probs"][wi]) if f is not None and f.get("probs") is not None else 1.0)
                if learn:
                    self._learn_slot(r["forward"], wi)
            else:
                r = self.speech.forward(s, group, gain, lateral)
                if r is None or r["probs"] is None:
                    break
                p = r["probs"]
                if temperature > 0:
                    rng = rng or np.random.default_rng()
                    q = p ** (1.0 / temperature)
                    w = self.name(int(rng.choice(len(q), p=q / q.sum())))
                else:
                    w = self.name(int(np.argmax(p)))
                confidence = min(confidence, float(p[self.known(w)]))
                for c, g in zip(r["active"], r["g"]):
                    involved[int(c)] = involved.get(int(c), 0.0) + float(g)
                if trace is not None:
                    raw = r["raw"] + lateral @ r["raw"]
                    top = np.argsort(-p)[:5]
                    trace.append([(self.name(i), float(raw[i] / raw.sum()), float(gain[i]), float(p[i])) for i in top])
            if w == END:
                break
            fatigue *= self.fatigue_decay
            fatigue[self.known(w)] = 1.0
            said.append(w)
            if target:
                self._joint_attention(scene, attention, w)
            if attention["at"] is not None:
                heard_by_object.setdefault(attention["at"], []).append(w)
                said_idx = [self.known(x) for x in heard_by_object[attention["at"]]]
                names = self._names(lifts)
                ov = self.overlap()
                # a name counts as said if a said word-form shares its sound
                if names and all(ov[n, said_idx].max() > 0.3 for n in names):
                    self._shift(scene, attention)

        if learn and target is not None:
            self._bind(scene, q_words, said, heard_by_object)
            for name, p in self.pathways.items():
                active = bool(scene["objects"])
                p.carry(1.0 if active else 0.0)
            ep_surprise = float(np.mean(surprises))
            self.surprise_trace = 0.99 * self.surprise_trace + 0.01 * ep_surprise
            for _, p in self.sprouted:
                if self.stable_sprouts:
                    # dopamine-gated: myelinate when things go better than usual
                    p.myelin = float(np.clip(p.myelin + 0.02 * (self.surprise_trace - ep_surprise), 0.0, 1.0))
                else:
                    p.carry(1.0)
            self.hippocampus.store((image, question, answer))
            self._monitor(float(np.mean(surprises)))
        self.now = None
        self.last_confidence, self.last_involved = confidence, involved
        return self.text(said), (correct / len(target) if target else None)

    # ------------------------------------------------- thinking against time

    def think(self, image, question, answer=None, reward=None, learn=True, max_tick=6):
        """Plan silently at each tick with what has arrived; speak when sure enough.

        The organism is never told how much time a question allows. It only
        gets `reward(tick, correct)` back after speaking (a dopamine-like
        scalar) and hears what the person would have said. Patience lives on
        the speech clusters that shaped the answer, and is tuned by outcome:
        right but late -> hastier; wrong after answering early -> more patient.
        Returns (answer, tick, reward).
        """
        scene = self.look(image, learn)
        for t in range(max_tick + 1):
            out, _ = self.live(image, question, learn=False, tick=t, scene=scene)
            involved = self.last_involved
            total = sum(involved.values())
            patience = sum(self.speech.patience[c] * g for c, g in involved.items()) / total if total else 1.0
            if self.last_confidence >= patience or t == max_tick:
                break
        got = None
        if answer is not None:
            correct = norm_text(out) == norm_text(answer)
            got = reward(t, correct) if reward else None
            if learn:
                self.live(image, question, answer, learn=True, tick=t, scene=scene)
                if got is not None:
                    self._tune_patience(involved, t, correct, got, max_tick)
        return out, t, got

    def _tune_patience(self, involved, t, correct, got, max_tick, eta=0.05):
        total = sum(involved.values()) or 1.0
        best_possible = 1.0 - self.time_cost * t
        if correct and got < best_possible - 1e-6:
            step = -eta  # right, but it cost me: hurry up
        elif not correct and t < max_tick:
            step = +eta  # spoke too soon: wait for more
        elif correct:
            step = -0.2 * eta  # fine: a little urgency is never wasted
        else:
            return
        for c, g in involved.items():
            if c < self.speech.C:
                self.speech.patience[c] = float(np.clip(self.speech.patience[c] + step * g / total, 0.02, 0.99))

    # ------------------------------------------------------------- learning

    def _bind(self, scene, q_words, said, heard_by_object):
        """Hebbian traces between what was seen and what was heard."""
        if not scene["objects"]:
            return
        V = self.V
        idx = lambda ws: [self.known(w) for w in ws]  # noqa: E731
        ov = self.overlap()

        def heard_vector(ws):
            """Heard forms, and every form sharing auditory neurons with them
            ("it is small" also excites the neurons of "small")."""
            return ov[:, idx(ws)].max(axis=1) if ws else np.zeros(V, np.float32)

        whole = heard_vector(said)  # what the speaker said about the scene
        # homeostasis: word neurons track their own average activity during vision
        self.visual_episodes += 1
        rate = max(0.003, 1.0 / self.visual_episodes)
        self.baseline[:V] += rate * (whole - self.baseline[:V])
        if scene["number"] is not None:
            self.areas["number"].bind(*scene["number"], whole)
        for a, v in scene.get("gist", {}).items():
            self.areas[a].bind(*v, whole)
        single = len(scene["objects"]) == 1
        for k, obj in enumerate(scene["objects"]):
            if single:
                heard = whole
            elif k in heard_by_object:
                heard = heard_vector(heard_by_object[k])
            else:
                continue
            for a in self.object_areas:
                self.areas[a].bind(*obj[a], heard)

    def _lateral(self):
        """Spread within emergent categories: a word that competes in the same
        slots as the active candidates gets some of their drive."""
        V = self.V
        A = self.assoc[:V, :V]
        return self.spread * A / np.maximum(1.0, A.sum(axis=1, keepdims=True))

    def _learn_slot(self, r, target, eta=0.05):
        """Words a familiar context expected, when another word came instead,
        are substitutable in that slot: strengthen their association."""
        if r is None or r.get("raw") is None or self.speech.theta_familiar is None:
            return
        if r["aff"][r["active"]].max() < self.speech.theta_familiar:
            return
        raw = r["raw"]
        total = raw.sum()
        if total <= 0:
            return
        expected = raw / total
        others = np.where(expected > 0.1)[0]
        others = others[others != target]
        for v in others:
            d = eta * expected[v]
            self.assoc[target, v] += d * (1 - self.assoc[target, v])
            self.assoc[v, target] += d * (1 - self.assoc[v, target])

    # ---------------------------------------------------------------- sleep

    def sleep(self, replay=300):
        dreams = self.hippocampus.replay(replay)
        for image, question, answer in dreams:
            self.live(image, question, answer, learn=True)
        died = self.speech.sleep(min_age=2000, merge_above=0.995)
        return len(dreams), died

    # ---------------------------------------------------------------- stats

    def report(self):
        areas = " ".join(f"{k}={a.C}" for k, a in self.areas.items())
        areas += "".join(f" {a.name}={a.size()}" for a, _ in self.sprouted)
        myel = " ".join(f"{k}={p.myelin:.2f}" for k, p in self.pathways.items())
        neurons = sum(a.neuron_count() for a in self.areas.values()) + self.speech.neuron_count() + self.V
        return (
            f"neurons~{neurons} | assemblies: {areas} speech-clusters={self.speech.C} "
            f"word-forms={self.V} | myelin: {myel}"
        )
