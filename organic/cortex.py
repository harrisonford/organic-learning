"""The cortex: a growing population of neuron clusters that learns without backprop.

Each cluster is a small group of neurons that shares an *identity* vector z.
Learning works like this, for each (stimulus, target) "problem":

1. Affinity: every cluster measures how close its identity is to the stimulus,
   aff_c = cos(z_c, s).
2. Problem affinity: a cluster has affinity to the *problem* when it is close
   to the stimulus AND its outgoing synapses already lean toward the target,
   P_c = aff_c * compat_c.
3. If no cluster has enough problem affinity and the organism did not answer
   confidently and correctly, neurogenesis: a new cluster is born, imprinted
   with the stimulus as its identity and wired to the target output neuron.
   The same happens when the answer was wrong because clusters that "know"
   the answer felt less familiar with the stimulus than the ones that voted
   wrong (match tracking, as in Grossberg's Adaptive Resonance Theory).
4. Otherwise only the specialists (clusters with problem affinity) change
   their weights, using local rules: instar (input weights move toward the
   stimulus), outstar (output weights move toward the observed outcome),
   identity drift, and a small exploratory noise term. The size of the change
   is scaled by a global "surprise" neuromodulator m = 1 - p(target), which is
   a coarse organism-level signal, not a per-weight gradient.
5. Slow structure: clusters that keep helping get myelinated (their output
   gain grows) and consolidated (their plasticity shrinks). Clusters that vote
   for wrong answers lose myelin. Co-active specialists form associations:
       dW_ij = eta * (alpha * a_i * a_j + beta * sim(z_i, z_j) - gamma * W_ij)
6. Sleep: weak, unused, demyelinated clusters die (apoptosis) and weak
   associations are pruned.

Nothing is ever differentiated; there is no backward pass.
"""

import numpy as np


class Cortex:
    def __init__(
        self,
        dim,
        neurons_per_cluster=8,
        active_k=8,
        theta_active=0.15,
        theta_problem=0.55,
        competition_temp=0.04,
        eta=0.25,
        identity_drift=0.15,
        noise=0.002,
        theta_familiar=None,
        associate=True,
        maturation=None,
        seed=0,
    ):
        self.dim = dim
        self.n = neurons_per_cluster
        self.active_k = active_k
        self.theta_active = theta_active
        self.theta_problem = theta_problem
        self.temp = competition_temp
        self.eta = eta
        self.drift = identity_drift
        self.noise = noise
        # a context this familiar may lead to new outcomes without growing a
        # new cluster: the cluster widens its expectations instead (None = off)
        self.theta_familiar = theta_familiar
        self.associate = associate  # Hebbian links between co-active clusters
        # synapses consolidate with experience: plasticity ~ 1 / (1 + uses / maturation),
        # so a mature cluster's expectations become a long-run average (None = off)
        self.maturation = maturation
        self.rng = np.random.default_rng(seed)

        self.C = 0  # number of living clusters
        self.cap = 64
        self.Z = np.zeros((self.cap, dim), np.float32)
        self.Win = np.zeros((self.cap, self.n, dim), np.float32)
        self.out_cap = 64
        self.Wout = np.zeros((self.cap, self.n, self.out_cap), np.float32)
        self.myelin = np.zeros(self.cap, np.float32)
        self.stability = np.zeros(self.cap, np.float32)
        self.usage = np.zeros(self.cap, np.int64)
        self.born = np.zeros(self.cap, np.int64)
        self.uid = np.zeros(self.cap, np.int64)  # stable id, survives compaction
        self.reach = np.zeros(self.cap, np.int64)  # how many inputs it has synapses on
        self.next_uid = 0

        self.outputs = []  # output neuron names, e.g. "vision:circle"
        self.out_index = {}
        self.assoc = {}  # uid_i -> {uid_j: association strength}

        self.t = 0
        self.births = 0
        self.deaths = 0

    def __setstate__(self, state):
        """Load brains saved before clusters had a reach (they reach everything)."""
        self.__dict__.update(state)
        if "reach" not in state:
            self.reach = np.full(self.cap, self.dim, np.int64)

    # ------------------------------------------------------------------ growth

    def output_neuron(self, name):
        """Return the index of an output neuron, growing one if it is new."""
        if name in self.out_index:
            return self.out_index[name]
        idx = len(self.outputs)
        if idx >= self.out_cap:
            self.out_cap *= 2
            grown = np.zeros((self.cap, self.n, self.out_cap), np.float32)
            grown[:, :, : self.Wout.shape[2]] = self.Wout
            self.Wout = grown
        self.outputs.append(name)
        self.out_index[name] = idx
        return idx

    def output_group(self, prefix):
        return np.array([i for i, o in enumerate(self.outputs) if o.startswith(prefix)], np.int64)

    def _grow_capacity(self):
        self.cap *= 2

        def grow(a):
            b = np.zeros((self.cap,) + a.shape[1:], a.dtype)
            b[: a.shape[0]] = a
            return b

        self.Z, self.Win, self.Wout = grow(self.Z), grow(self.Win), grow(self.Wout)
        self.myelin, self.stability = grow(self.myelin), grow(self.stability)
        self.usage, self.born, self.uid = grow(self.usage), grow(self.born), grow(self.uid)
        self.reach = grow(self.reach)

    def grow_inputs(self, extra):
        """New afferent axons arrive: every cluster gets `extra` silent synapses."""
        pad = lambda a, axis: np.concatenate([a, np.zeros(a.shape[:axis] + (extra,) + a.shape[axis + 1 :], a.dtype)], axis=axis)  # noqa: E731
        self.Z = pad(self.Z, 1)
        self.Win = pad(self.Win, 2)
        self.dim += extra

    def neurogenesis(self, s, target):
        """Birth of a new cluster, imprinted with the stimulus and the target."""
        if self.C >= self.cap:
            self._grow_capacity()
        c = self.C
        self.Z[c] = s
        jitter = self.rng.normal(0, 0.15 / np.sqrt(self.dim), (self.n, self.dim))
        w = s[None, :] + jitter
        self.Win[c] = w / np.linalg.norm(w, axis=1, keepdims=True)
        self.Wout[c] = 0.0
        self.Wout[c, :, target] = 1.0
        self.myelin[c] = 0.5
        self.stability[c] = 0.0
        self.usage[c] = 1
        self.born[c] = self.t
        self.uid[c] = self.next_uid
        self.reach[c] = self.dim
        self.next_uid += 1
        self.C += 1
        self.births += 1
        return c

    # ----------------------------------------------------------------- sensing

    def _seen(self, s, r):
        """The part of the stimulus a cluster with reach r has synapses for."""
        if r == len(s):
            return s
        part = s[:r]
        n = np.linalg.norm(part)
        return np.concatenate([part / n if n > 0 else part, np.zeros(len(s) - r, s.dtype)])

    def _affinity(self, s, idx=None):
        idx = np.arange(self.C) if idx is None else idx
        aff = np.empty(len(idx), np.float32)
        reach = self.reach[idx]
        for r in np.unique(reach):
            m = reach == r
            aff[m] = self.Z[idx[m]] @ self._seen(s, r)
        return aff

    def _cluster_activity(self, idx, s):
        """Neuron activity inside clusters: rectified match, k-winners-take-all."""
        a = np.zeros((len(idx), self.n), np.float32)
        reach = self.reach[idx]
        for r in np.unique(reach):
            m = reach == r
            a[m] = np.maximum(self.Win[idx[m]] @ self._seen(s, r), 0.0)
        if a.size:
            kth = np.partition(a, self.n // 2, axis=1)[:, self.n // 2][:, None]
            a = np.where(a >= kth, a, 0.0)
            a = a / (a.max(axis=1, keepdims=True) + 1e-8)
        return a

    def forward(self, s, group, gain=None, lateral=None):
        """Let activity flow from the stimulus to the output neurons of a group.

        gain: optional multiplicative modulation of each output neuron in the
        group (priming from other areas, fatigue). It changes which neuron wins
        without touching any synapse.
        lateral: optional (len(group), len(group)) spread of activation between
        output neurons before gating.
        """
        if self.C == 0 or len(group) == 0:
            return None
        aff = self._affinity(s)
        # spreading activation through learned associations between clusters
        k = min(self.active_k, self.C)
        top = np.argpartition(-aff, k - 1)[:k]
        top = top[aff[top] > self.theta_active]
        if len(top) == 0:
            return {"aff": aff, "active": top, "probs": None}
        if self.assoc:
            uid_to_c = {int(self.uid[c]): c for c in range(self.C)}
            bonus = {}
            for c in top:
                for j, w in self.assoc.get(int(self.uid[c]), {}).items():
                    if j in uid_to_c:
                        bonus[uid_to_c[j]] = bonus.get(uid_to_c[j], 0.0) + w * aff[c]
            if bonus:
                aff = aff.copy()
                for c, b in bonus.items():
                    aff[c] += 0.1 * b
                top = np.unique(np.concatenate([top, np.fromiter(bonus.keys(), np.int64)]))
                top = top[aff[top] > self.theta_active]

        a = self._cluster_activity(top, s)
        # lateral inhibition: the closest clusters dominate the vote
        g = np.exp((aff[top] - aff[top].max()) / self.temp)
        g = g / g.sum()
        votes = np.einsum("kn,kno->ko", a, self.Wout[top][:, :, group])
        votes = votes / (a.sum(axis=1, keepdims=True) + 1e-8)  # per-cluster output tendency
        o = ((g * self.myelin[top])[:, None] * votes).sum(axis=0)
        raw = o
        if lateral is not None:
            o = o + lateral @ o
        if gain is not None:
            o = o * gain
        probs = o / o.sum() if o.sum() > 0 else np.full(len(group), 1.0 / len(group))
        return {"aff": aff, "active": top, "a": a, "g": g, "votes": votes, "probs": probs, "raw": raw}

    # ---------------------------------------------------------------- learning

    def learn(self, s, target, group, gain=None, lateral=None):
        """Experience one problem (stimulus s should evoke output neuron `target`)."""
        self.t += 1
        gpos = int(np.where(group == target)[0][0])
        r = self.forward(s, group, gain, lateral)

        if r is None or r["probs"] is None:
            self.neurogenesis(s, target)
            return {"grew": True, "correct": False, "forward": r}

        probs = r["probs"]
        correct = int(np.argmax(probs)) == gpos
        surprise = 1.0 - float(probs[gpos])  # global neuromodulator

        # problem affinity of every reasonably close cluster
        # P = aff * compat with compat <= 1, so only clusters at least this close
        # to the stimulus can be specialists (or familiar)
        floor = max(self.theta_active, min(self.theta_problem, self.theta_familiar or 1.0))
        cand = np.where(r["aff"][: self.C] >= floor)[0]
        a_c = self._cluster_activity(cand, s)
        compat = (a_c * self.Wout[cand, :, target]).sum(axis=1) / (a_c.sum(axis=1) + 1e-8)
        P = r["aff"][cand] * compat
        chosen = P >= self.theta_problem
        if self.theta_familiar is not None:
            familiar = r["aff"][cand] >= self.theta_familiar
            P = np.where(familiar, np.maximum(P, r["aff"][cand]), P)
            chosen |= familiar
        specialists = cand[chosen]

        # match tracking: the stimulus felt more familiar to clusters that got it
        # wrong than to any cluster that knows the answer -> grow a finer one
        outcompeted = (
            not correct
            and len(specialists) > 0
            and r["aff"][specialists].max() < r["aff"][r["active"]].max() - 1e-6
        )
        grew = False
        if (len(specialists) == 0 and (not correct or surprise > 0.5)) or outcompeted:
            self.neurogenesis(s, target)
            grew = True
        if len(specialists) > 0:
            spec_a = a_c[chosen]
            spec_P = P[chosen]
            outcome = np.zeros(len(group), np.float32)
            outcome[gpos] = 1.0
            for c, a, p in zip(specialists, spec_a, spec_P):
                s_c = self._seen(s, self.reach[c])
                lr = self.eta * p * (1.0 - self.stability[c]) * (0.2 + surprise)
                if self.maturation:
                    lr = max(lr / (1.0 + self.usage[c] / self.maturation), 0.01)
                z = self.Z[c] + lr * self.drift * (s_c - self.Z[c])
                self.Z[c] = z / np.linalg.norm(z)
                # instar: active neurons' input weights move toward the stimulus
                self.Win[c] += lr * a[:, None] * (s_c[None, :] - self.Win[c])
                rc = self.reach[c]
                self.Win[c][:, :rc] += self.noise * self.rng.standard_normal((self.n, rc))
                # outstar: active neurons' output weights move toward what happened
                wo = self.Wout[c][:, group]
                self.Wout[c][:, group] = wo + lr * a[:, None] * (outcome[None, :] - wo)
                self.usage[c] += 1
                if correct:
                    self.myelin[c] = min(2.0, self.myelin[c] + 0.02)
                    self.stability[c] = min(0.9, self.stability[c] + 0.003)
            self._associate(specialists, spec_a)

        # clusters that voted strongly for the wrong answer lose some myelin
        if not correct:
            wrong = r["votes"].argmax(axis=1) != gpos
            for c, w, gi in zip(r["active"], wrong, r["g"]):
                if w and gi > 0.2:
                    self.myelin[c] = max(0.05, self.myelin[c] * 0.9)

        return {"grew": grew, "correct": correct, "forward": r}

    def _associate(self, clusters, acts, alpha=1.0, beta=0.5, gamma=0.05, eta=0.05):
        """Hebbian association between co-active specialist clusters."""
        if len(clusters) < 2 or not self.associate:
            return
        strength = acts.mean(axis=1)
        for x in range(len(clusters)):
            for y in range(len(clusters)):
                if x == y:
                    continue
                i, j = int(self.uid[clusters[x]]), int(self.uid[clusters[y]])
                sim = float(self.Z[clusters[x]] @ self.Z[clusters[y]])
                row = self.assoc.setdefault(i, {})
                w = row.get(j, 0.0)
                w += eta * (alpha * strength[x] * strength[y] + beta * sim - gamma * w)
                row[j] = min(w, 1.0)

    # ------------------------------------------------------------------- sleep

    def sleep(self, min_age=200, myelin_floor=0.2, merge_above=None):
        """Consolidation, apoptosis of useless clusters, pruning of weak associations.

        merge_above: clusters whose identities are this similar are fused into
        one (the more used one absorbs the other), so redundant copies of the
        same memory collapse into a single, more general one.
        """
        merged = self._merge(merge_above) if merge_above else np.zeros(self.C, bool)
        age = self.t - self.born[: self.C]
        dead = merged | (age > min_age) & (self.myelin[: self.C] < myelin_floor) & (self.usage[: self.C] <= 2)
        keep = np.where(~dead)[0]
        died = int(dead.sum())
        if died:
            dead_uids = set(self.uid[np.where(dead)[0]].tolist())
            for arr in ("Z", "Win", "Wout", "myelin", "stability", "usage", "born", "uid", "reach"):
                a = getattr(self, arr)
                a[: len(keep)] = a[keep]
            self.C = len(keep)
            self.deaths += died
        else:
            dead_uids = set()
        self.assoc = {
            i: {j: w * 0.9 for j, w in row.items() if w > 0.05 and j not in dead_uids}
            for i, row in self.assoc.items()
            if i not in dead_uids
        }
        return died

    def _merge(self, threshold):
        C = self.C
        absorbed = np.zeros(C, bool)
        if C < 2:
            return absorbed
        sim = self.Z[:C] @ self.Z[:C].T
        np.fill_diagonal(sim, -1)
        order = np.argsort(-self.usage[:C])  # most used clusters absorb first
        for i in order:
            if absorbed[i]:
                continue
            mates = np.where((sim[i] > threshold) & ~absorbed & (self.reach[:C] == self.reach[i]))[0]
            mates = mates[mates != i]
            for j in mates:
                if self.usage[j] > self.usage[i]:
                    continue
                wi, wj = float(self.usage[i]), float(self.usage[j])
                f = wj / (wi + wj)
                z = (1 - f) * self.Z[i] + f * self.Z[j]
                self.Z[i] = z / np.linalg.norm(z)
                self.Win[i] = (1 - f) * self.Win[i] + f * self.Win[j]
                self.Wout[i] = (1 - f) * self.Wout[i] + f * self.Wout[j]
                self.myelin[i] = max(self.myelin[i], self.myelin[j])
                self.usage[i] += self.usage[j]
                absorbed[j] = True
        return absorbed

    # ------------------------------------------------------------------- stats

    def neuron_count(self):
        return self.C * self.n
