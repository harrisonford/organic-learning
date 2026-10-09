# organic-learning

An experiment in learning **without backpropagation**. A single cortex grows
neurons when it meets problems it has no affinity for. Only the neurons that
have affinity for a problem change their weights, and they change them with
local rules. Optimization quality is *not* the goal yet. The goal is to be as
organic as possible and see what emerges.

```
 image ──► VisionOrgan  ─┐   (fixed retina: blur + oriented edges)
                         ├──► sensory sheet ──► Cortex (growing clusters) ──► output neurons
 text  ──► LanguageOrgan ┘   (fixed word "sounds", fading echo of last words)       vision:circle, language:hello, ...
```

## The paradigm

- **Cluster = a small set of neurons** with a shared *identity* vector `z`,
  per-neuron input and output synapses, *myelin* (output gain / trust),
  *stability* (consolidation), and a usage history.
- **Affinity to the stimulus**: `aff_c = cos(z_c, s)`.
- **Affinity to the problem**: `P_c = aff_c · compat_c`, where `compat_c` is
  how much the cluster's outgoing synapses already lean toward the observed
  outcome.
- **Neurogenesis**: a new cluster is born, imprinted with the stimulus and
  wired to the outcome, in two cases. Either no cluster has enough problem
  affinity and the organism didn't answer confidently and correctly, or it
  answered wrong because clusters that know the answer felt *less familiar*
  with the stimulus than the ones that voted wrong (ART-style match tracking).
- **Affinity-gated local learning**: only the specialists (`P_c ≥ θ`) change.
  - instar: `ΔW_in = lr · a · (s − W_in)`
  - outstar: `ΔW_out = lr · a · (outcome − W_out)`
  - identity drift toward the stimuli the cluster serves, plus a small amount of
    exploratory noise
  - `lr = η · P_c · (1 − stability_c) · (0.2 + m)`, where `m = 1 − p(target)`
    is a single, global "surprise" neuromodulator. No per-weight error, no
    chain rule.
- **Myelination**: clusters that take part in correct answers gain myelin
  (louder vote) and stability (less plastic). Clusters that vote strongly for
  wrong answers lose myelin.
- **Associations** between co-active specialists:
  `ΔA_ij = η(α a_i a_j + β sim(z_i, z_j) − γ A_ij)`. They spread activation
  to related clusters.
- **Sleep**: old, unused, demyelinated clusters die (apoptosis), and weak
  associations decay and are pruned.
- **Organs** are hard-wired and never trained. Each writes into its own region
  of the sensory sheet and owns a group of output neurons that grows when it
  meets a new label or word. The cortex is never told which region is which:
  vision and language *territories* emerge from where clusters' identities
  settle.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python examples/demo.py
```

The demo raises one organism on interleaved experience: 900 noisy 16×16 shapes
(circle, square, triangle, cross, hline, vline) and 34 short dialogues. Each
"day" ends with sleep. Then it tests the organism on 300 unseen shapes and a
set of prompts, some of which it has never heard. It writes
`out/vision_predictions.png`.

Typical result (about 4 s on CPU):

```
day 4: vision online acc=0.89
   neurons=2976 clusters=372 ... | territories: vision=1096, language=1880, mixed=0
accuracy 0.87 (chance 0.17)
  you: hey , how are you ?      org: i am fine , thank you
  you: what color is grass ?    org: the grass is green
  you: what do dogs say ?       org: cats say meow        <- confuses similar contexts
  you: do you like dogs ?       org: yes , i like music very much
```

## Honest limitations / next directions

- Language is mostly associative recall of the contexts it has heard. It
  generalizes only by similarity between word-echo vectors, and nothing
  composes meaning.
- Vision relies on near-prototype matching, so it is not translation-invariant
  beyond what the pooled edge features give it.
- The cortex is flat: a cluster maps sensations directly to outputs. Next
  steps could be a hierarchy of clusters feeding clusters, conduction delays /
  timing (the myelination-as-timing idea), cross-modal association (seeing a
  circle while hearing "circle"), and neuron-level growth inside clusters.
