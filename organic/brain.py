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
from .ear import Ear
from .eye import Eye
from .hippocampus import Hippocampus
from .pathway import Pathway

END = "<end>"


class Brain:
    def __init__(self, seed=0, lam=1.5, spread=0.3, fatigue=0.9, coincidence_weight=0.8, category_blend=0.5, rem_weight=0.6):
        self.eye = Eye()
        self.ear = Ear(seed=seed + 7)
        self.areas = {
            "shape": SensoryArea("shape", Eye.form_dim, vigilance=0.86),
            "color": SensoryArea("color", Eye.color_dim, vigilance=0.9),
            "where": SensoryArea("where", Eye.where_dim, vigilance=0.88),
            "number": SensoryArea("number", Eye.number_dim, vigilance=0.9),
        }
        self.object_areas = ("shape", "color", "where")
        self.pathways = {k: Pathway(f"{k}->speech") for k in self.areas}
        self.co_dim = 16
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
        self.co_weight = coincidence_weight
        self.category_blend = category_blend
        self.rem_weight = rem_weight
        self.vcap = 64
        self.baseline = np.zeros(self.vcap, np.float32)  # homeostatic average activity
        self.assoc = np.zeros((self.vcap, self.vcap), np.float32)  # word-word slot associations
        self.visual_episodes = 0
        self.age = 0

    # -------------------------------------------------------------- lexicon

    def word(self, w):
        i = self.speech.output_neuron(f"word:{w}")
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
        return self.speech.outputs[i].split(":", 1)[1]

    # ----------------------------------------------------------- perception

    def look(self, image, learn):
        """Saccade over the scene and let the sensory areas respond."""
        if image is None:
            return {"fixations": [], "objects": [], "number": None}
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
        return {"fixations": fix, "objects": objects, "number": number}

    def lifts(self, scene, attended):
        """Per-area lift over the lexicon, as delivered by each pathway."""
        V = self.V
        k = 0.02
        out = {}
        sources = {}
        if scene["objects"] and attended is not None:
            for a in self.object_areas:
                sources[a] = scene["objects"][attended][a]
        if scene["number"] is not None:
            sources["number"] = scene["number"]
        for a, (idx, act) in sources.items():
            e = self.areas[a].expectation(idx, act, V)
            if e is None:
                continue
            lift = (e + k) / (self.baseline[:V] + k)
            out[a] = np.clip(lift, 0.05, 10.0)
        return out

    def priming(self, lifts):
        """Facilitation of word neurons by each area, scaled by how much of the
        signal arrives in time. Priming only excites: an area that has never
        seen a white circle does not veto "circle"; suppression of the other
        candidates comes from competition between word neurons."""
        g = np.ones(self.V, np.float32)
        for a, lift in lifts.items():
            g *= np.maximum(lift, 1.0) ** (self.lam * self.pathways[a].gain)
        return g

    def coincidence(self, q_words, lifts):
        """Neurons that fire when a heard word is also one vision is priming."""
        v = np.zeros(self.co_dim, np.float32)
        if not lifts:
            return v
        total = np.ones(self.V, np.float32)
        for a, lift in lifts.items():
            total *= np.maximum(lift, 1.0) ** self.pathways[a].gain
        known = [self.speech.out_index[f"word:{w}"] for w in q_words if f"word:{w}" in self.speech.out_index]
        if not known:
            return v
        m = float(np.log(total[known].max()))
        centers = np.linspace(0, 4, self.co_dim)
        return np.exp(-0.5 * ((m - centers) / 0.45) ** 2).astype(np.float32)

    def inner_voice(self, w):
        """How my own word feels: its sound blended with the sounds of the words
        it substitutes for, so a word carries a sense of its emergent category."""
        v = self.ear.sound(w)
        i = self.speech.out_index.get(f"word:{w}")
        if i is None or i >= self.V:
            return v
        row = self.assoc[i, : self.V]
        total = row.sum()
        if total <= 0:
            return v
        others = sum(row[j] * self.ear.sound(self.name(j)) for j in np.nonzero(row > 0.05)[0])
        blend = v + self.category_blend * others / max(total, 1.0)
        return blend / np.linalg.norm(blend) * np.linalg.norm(v)

    def speech_input(self, q_code, said, co, remaining):
        s = np.concatenate([q_code, self.ear.said(said, self.inner_voice), self.co_weight * co, self.rem_weight * remaining])
        return s / np.linalg.norm(s)

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

    def _joint_attention(self, scene, attention, w):
        """While listening, look at the object that the heard word is about."""
        if len(scene["objects"]) < 2 or f"word:{w}" not in self.speech.out_index:
            return False
        wi = self.speech.out_index[f"word:{w}"]
        best, best_lift = None, 2.0
        for k in range(len(scene["objects"])):
            if k in attention["visited"]:
                continue
            ls = self.lifts(scene, k)
            lift = max(ls[a][wi] for a in ("shape", "color") if a in ls)
            if lift > best_lift:
                best, best_lift = k, lift
        if best is not None and best != attention["at"]:
            self._shift(scene, attention, to=best)
            return True
        return False

    def live(self, image, question, answer=None, learn=True, temperature=0.0, max_words=16, rng=None, trace=None):
        """Experience a moment. With an answer: imitate it. Without: respond.

        trace: optional list; each speaking step appends its top candidates.
        """
        self.age += 1
        scene = self.look(image, learn)
        q_words = self.ear.words(question)
        for w in q_words:
            self.word(w)
        target = None
        if answer is not None:
            target = self.ear.words(answer) + [END]
            for w in target:
                self.word(w)
        if learn:
            self.ear.listen(q_words)
            if target:
                self.ear.listen(target)
        q_code = self.ear.question(q_words)
        group = self.lexicon()
        fatigue = np.zeros(self.vcap, np.float32)
        for w in q_words:  # just-heard words are a little fatigued too
            fatigue[self.speech.out_index[f"word:{w}"]] = 0.3
        attention = {"at": 0 if scene["objects"] else None, "visited": set()}
        said, heard_by_object = [], {}
        correct = 0
        steps = len(target) if target else max_words
        for t in range(steps):
            lifts = self.lifts(scene, attention["at"])
            prime = self.priming(lifts)
            co = self.coincidence(q_words, lifts)
            s = self.speech_input(q_code, said, co, self._remaining_code(scene, attention))
            gain = self.word_gain(prime, fatigue[: self.V])
            lateral = self._lateral()
            if target:
                w = target[t]
                wi = self.speech.out_index[f"word:{w}"]
                r = self.speech.learn(s, wi, group, gain, lateral)
                correct += r["correct"]
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
                if trace is not None:
                    raw = r["raw"] + lateral @ r["raw"]
                    top = np.argsort(-p)[:5]
                    trace.append([(self.name(i), float(raw[i] / raw.sum()), float(gain[i]), float(p[i])) for i in top])
            if w == END:
                break
            fatigue *= 0.4
            fatigue[self.speech.out_index[f"word:{w}"]] = 1.0
            said.append(w)
            if target:
                self._joint_attention(scene, attention, w)
            if attention["at"] is not None:
                heard_by_object.setdefault(attention["at"], []).append(w)
                said_idx = {self.speech.out_index[f"word:{x}"] for x in heard_by_object[attention["at"]]}
                names = self._names(lifts)
                if names and all(n in said_idx for n in names):
                    self._shift(scene, attention)

        if learn and target is not None:
            self._bind(scene, q_words, said, heard_by_object)
            for name, p in self.pathways.items():
                active = name == "number" and scene["number"] is not None or (name != "number" and scene["objects"])
                p.carry(1.0 if active else 0.0)
            self.hippocampus.store((image, question, answer))
        return " ".join(said), (correct / len(target) if target else None)

    # ------------------------------------------------------------- learning

    def _bind(self, scene, q_words, said, heard_by_object):
        """Hebbian traces between what was seen and what was heard."""
        if not scene["objects"]:
            return
        V = self.V
        idx = lambda ws: [self.speech.out_index[f"word:{w}"] for w in ws]  # noqa: E731
        whole = np.zeros(V, np.float32)
        whole[idx(said)] = 1.0  # what the speaker said about the scene
        # homeostasis: word neurons track their own average activity during vision
        self.visual_episodes += 1
        rate = max(0.003, 1.0 / self.visual_episodes)
        self.baseline[:V] += rate * (whole - self.baseline[:V])
        if scene["number"] is not None:
            self.areas["number"].bind(*scene["number"], whole)
        single = len(scene["objects"]) == 1
        for k, obj in enumerate(scene["objects"]):
            if single:
                heard = whole
            elif k in heard_by_object:
                heard = np.zeros(V, np.float32)
                heard[idx(heard_by_object[k])] = 1.0
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
        myel = " ".join(f"{k}={p.myelin:.2f}" for k, p in self.pathways.items())
        neurons = sum(a.neuron_count() for a in self.areas.values()) + self.speech.neuron_count() + self.V
        return (
            f"neurons~{neurons} | assemblies: {areas} speech-clusters={self.speech.C} "
            f"words={self.V} | myelin: {myel}"
        )
