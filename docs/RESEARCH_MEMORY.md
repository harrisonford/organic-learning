# Memory: what this brain lacks (literature survey, 2026-11-17)

Asked by Kris on 2026-11-17 ("we could delve into memory also, maybe that could be a missing piece"). A survey by a
research agent, kept nearly as returned. Tags: **[V]** checked against a source in that session, **[M]** recalled
from the literature and not re-checked, **[I]** the surveyor's inference. Nothing here is measured in this project
yet; the experiments at the end are the plan.

## Diagnosis

The brain has slow "cortical" weights (word cells, clusters) and prototype clusters. It lacks:

- a fast, one-shot, content-addressable store of episodes;
- any state that outlives a moment (every moment starts blank);
- consolidation tied to how reliable a memory is (it counts use instead).

**Check first [I]:** the fall of COCO naming from 0.32 to 0.22 after its lessons stopped may be a failure to
retrieve, not erasure. Force the recall route on those pictures: if naming returns, the orchestrator's choice of
route (or the ear's units) drifted, and the fix belongs there and not in the word cells.

## 1. Complementary learning systems

- A hippocampal store writes each episode once under a sparse, decorrelated key (dentate gyrus) and completes
  partial cues (CA3) [M]. The cortex learns slowly from replay interleaved with new experience [M].
- Interleaving only old items that resemble the new ones is enough (Saxena et al. 2022) [V].
- Human rest replay prioritises weakly learned items, those most at risk (Schapiro et al. 2018) [V].
- Alternating hippocampus-driven replay of recent items with cortex-generated replay of remote ones protects old
  knowledge; blocked replay does not (Singh et al. 2022; contrastive Hebbian, no backprop) [V].
- Replaying internal representations scales better than raw inputs (van de Ven et al. 2020) [V].
- **Why our replay gave +0.02 [I]:** surprise picks new or noisy moments, not old items at risk; sleep replay comes
  after the damage and not interleaved with the lessons that interfere; a delta rule gains nothing from items it
  already gets right.

## 2. One-shot plasticity

- Wu & Maass 2025 [V]: binary weights; per item each memory neuron gets a plateau with probability about 0.005, and
  in those neurons synapses from active inputs flip with probability 0.5; recall is one thresholded pass. 30,000
  items in a 25,000-input, 39,000-neuron network, recalled with a third of the cue masked; similar items get
  repelled traces (pattern separation).
- Novelty drives locus coeruleus dopamine, which makes weak hippocampal memories persist, also when the novelty
  comes after encoding (Takeuchi et al. 2016; Tse et al. 2023) [V]. Tag and capture (Frey & Morris) [M].
- **For us [I]:** "learned fast only if the cue is new" is the delta rule's blocking: a cue that already predicts
  leaves no error. A store that writes on the novelty of the conjunction (low best match) does not have it.

## 3. Dense associative memory, sparse distributed memory

- `h = softmax(beta * K @ q); y = V @ h`, keys as rows of K, values as columns of V.
- Local rules (Tyulmankov et al. 2021) [V]: the key is written into one slot's input weights, gated by a local
  dendritic factor and a global modulator; the value by a Hebbian rule. Capacity near 1.0 N against 0.14 N for a
  classical Hopfield network.
- Modern Hopfield capacity is exponential in dimension with one-step retrieval, and has metastable states that
  average similar items (Ramsauer et al. 2020) [V]: low beta blends episodes into a prototype, high beta returns
  one episode [I].
- Kanerva's memory: fixed random addresses, writes to all addresses within a radius; attention approximates it
  (Bricken & Pehlevan 2021) [V].
- **For us [I]:** key = a sparse random expansion of the visual code, the question's units and the context; value =
  the adult's reply units in order (and the route taken, the reward); written at the end of a moment when novelty
  is high; read at question time as an extra cue on the content dendrite and as a further way of speaking.

## 4. Working memory and the thread of a conversation

- Activity-silent memory: presynaptic facilitation holds an item for about a second and a nonspecific input
  reactivates it (Mongillo et al. 2008) [V].
- Fast weights (Ba et al. 2016) [V]: `A = lam*A + eta*outer(h, h)`, used as an extra input `A @ h`; the rule is a
  local outer product.
- Prefrontal gating (O'Reilly & Frank 2006): slots held or overwritten by a gate learned from reward [V/M].
- Temporal context model (Howard & Kahana 2002) [V]: a context vector drifts slowly; retrieving an item brings
  back the context it was stored in.
- **For us [I]:** do not reset between moments: keep `c = rho*c + (1-rho)*x` at two or three rates, feed it to the
  context dendrite and include it in episodic keys. "How are you?" → "you?" is the copy route winning because no
  stored reply exists.

## 5. Episodic control

- A table keyed by a random projection of the state stores the best return per action, read by nearest neighbours
  (Blundell et al. 2016; Pritzel et al. 2017) [V]. It wins when data are scarce (Lengyel & Dayan 2007) [M].
- **For us [I]:** the clusters with exemplars already do this for gaze and route; the exemplars carry no content
  (no reply).

## 6. Forgetting without replay

- Benna & Fusi 2016: each synapse a chain of coupled variables, fast to slow [V].
- Synaptic intelligence accumulates importance online at each synapse [M]. Our 1/(1+use/5) counts use, not
  contribution [I].
- Recall-gated consolidation (Lindsey & Litwin-Kumar 2024) [V]: a fast system learns everything; the slow system
  updates only when the fast system's recall already agrees. Raises signal to noise and gives spacing effects.
- Sparse expansion with a perceptron-like rule reduces forgetting (Shen et al. 2023; Bricken et al. 2023) [V], at
  some expected cost in generalisation [I].

## 7. Development (so the mechanism matches the stage)

- Retention grows with age; brief reminders reactivate lapsed memories (Rovee-Collier) [V].
- Ordered recall is fragile at 9 months, reliable after 13, one exposure enough by 14 (Bauer) [V].
- The hippocampus encodes single events from about 12 months (Yates et al. 2025) [V]; the circuit reaches a
  milestone at 18 to 24 months (Gómez & Edgin 2016) [V].
- At 24 months a fast-mapped word is at chance after five minutes unless named ostensively and repeatedly (Horst &
  Samuelson 2008) [V]. Naps produce word generalisation at 9 to 16 months, scaling with spindles (Friedrich et al.
  2015) [V].
- **So [I]:** at two words the right store is small, leaky and reminder-dependent: one-shot traces that die unless
  repeated or replayed, with capacity and retention maturing with the stage.

## Ranked additions, each with a cheap offline test (the ordering is the surveyor's inference)

| # | addition | helps | offline test |
|---|---|---|---|
| 1 | sparse-keyed episodic store, novelty-gated write, read as a cue and as a way of speaking | few exposures, context | on one-exposure lessons, recall by the store against the word cells, old and new cues apart |
| 2 | context that is not reset between moments (two or three rates), gated slots | context | decode the previous turn's topic with and without reset; two-turn probes |
| 3 | replay of what is at risk and of what resembles the current lesson, interleaved while awake | forgetting | teach A, then 30,000 moments of B; none / uniform / surprise / falling margin / similar-to-current at equal budget |
| 4 | fast and slow weight per synapse, transfer gated by recall | forgetting | word cells alone on the same A-then-B schedule |
| 5 | sparse expansion before the content dendrite | forgetting | sweep expansion and sparsity; retention and held-out naming together |
| 6 | conjunction-novelty gate with tag and capture | few exposures | one exposure with an old cue, tested at once and after 1,000 moments |

## References

McClelland, McNaughton & Lampinen 2020 https://escholarship.org/uc/item/51d2044x ·
Kumaran, Hassabis & McClelland 2016 https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf ·
Saxena, Shobe & McNaughton 2022 https://escholarship.org/uc/item/1rw1b1m4 ·
Schapiro et al. 2018 https://pmc.ncbi.nlm.nih.gov/articles/PMC6156217 ·
Singh, Norman & Schapiro 2022 https://pmc.ncbi.nlm.nih.gov/articles/PMC9636926 ·
van de Ven, Siegelmann & Tolias 2020 https://doi.org/10.1038/s41467-020-17866-2 ·
Wu & Maass 2025 https://www.biorxiv.org/content/10.1101/2023.04.04.535572.full ·
Takeuchi et al. 2016 https://link.springer.com/10.1038/nature19325 ·
Tse et al. 2023 https://pmc.ncbi.nlm.nih.gov/articles/PMC10655220 ·
Ramsauer et al. 2020 https://arxiv.org/abs/2008.02217 ·
Krotov & Hopfield 2021 https://arxiv.org/abs/2008.06996 ·
Bricken & Pehlevan 2021 https://arxiv.org/abs/2111.05498 ·
Bricken et al. 2023 https://arxiv.org/abs/2303.11934 ·
Tyulmankov et al. 2021 https://arxiv.org/abs/2110.13976 ·
Mongillo, Barak & Tsodyks 2008 https://doi.org/10.1126/science.1150769 ·
Ba et al. 2016 https://arxiv.org/abs/1610.06258 ·
O'Reilly & Frank 2006 https://ccnlab.org/papers/HazyFrankOReilly06.pdf ·
Howard & Kahana 2002 https://people.bu.edu/marc777/docs/HowardKahana02-JMP.pdf ·
Blundell et al. 2016 https://arxiv.org/abs/1606.04460 ·
Pritzel et al. 2017 https://arxiv.org/abs/1703.01988 ·
Benna & Fusi 2016 https://doi.org/10.1038/nn.4401 ·
Lindsey & Litwin-Kumar 2024 https://elifesciences.org/articles/90793 ·
Shen, Dasgupta & Navlakha 2023 https://arxiv.org/abs/2107.07617 ·
Horst & Samuelson 2008 https://research-portal.uea.ac.uk/en/publications/fast-mapping-but-poor-retention-by-24-month-old-infants/ ·
Friedrich et al. 2015 https://pmc.ncbi.nlm.nih.gov/articles/PMC4316748/ ·
Yates et al. 2025 https://doi.org/10.1126/science.adt7570 ·
Gómez & Edgin 2016 https://pmc.ncbi.nlm.nih.gov/articles/PMC4808499 ·
Bauer & Lukowski 2010 https://faculty.sites.uci.edu/lukowski/files/2019/08/Bauer-Lukowski-2010.pdf ·
Rovee-Collier (review) https://pmc.ncbi.nlm.nih.gov/articles/PMC5716629

Cited from memory only: Bittner 2017, Milstein 2021, Frey & Morris 1997, Kanerva 1988, Krotov & Hopfield 2016,
Lengyel & Dayan 2007, Fusi et al. 2005, Kirkpatrick 2017, Zenke 2017, Carey & Bartlett 1978.

## The check, done (2026-11-17, `experiments/forget_diag.py`, seed 0, exact names on 80 held-out pictures a source)

| brain | source | the brain names it | the word cells alone | spoke by recall |
|---|---|---|---|---|
| world3_s0 at 60,000 (COCO lessons stopped at 50,000) | COCO things | 0.12 | 0.12 | 0.96 |
| | CIFAR | 0.50 | 0.50 | 0.97 |
| | Commons | 0.59 | 0.59 | 1.00 |
| world3_s0 at 125,000 | COCO things | 0.16 | 0.17 | 0.97 |
| | CIFAR | 0.45 | 0.49 | 0.95 |
| | Commons | 0.54 | 0.57 | 0.94 |

The brain speaks by recall on 95% of pictures and says what the word cells say. So the loss is not a failure to
retrieve and not the choice of route: **what was lost was lost in the word cells' synapses**, within 10,000 moments of
the lessons stopping. The additions against forgetting (3 to 5 in the table above) are the ones this points to.

## Dendritic branches in the word cells: negative (2026-11-17, `experiments/branch_probe.py`, three seeds)

Idea: a cell's visual synapses on several branches, the cell driven by its best branch, only that branch learning,
and a look of its word that drives it less than half its usual level taking an unused branch (branch-specific
plasticity, Cichon & Gan 2015). So that a new look of a known word need not overwrite the old one.

Word cells alone on the old cortex: 6,000 naming moments on three sources, then 6,000 without COCO things.

| held-out naming, after phase 1 → after phase 2 | CIFAR | Commons | COCO things | COCO things whose names go on being taught |
|---|---|---|---|---|
| one branch, seed 0 / 1 / 2 | 0.395 → 0.520 / 0.370 → 0.470 / 0.380 → 0.440 | 0.450 → 0.520 / 0.525 → 0.565 / 0.440 → 0.470 | 0.315 → 0.275 / 0.225 → 0.220 / 0.295 → 0.250 | 0.348 → 0.304 / 0.249 → 0.243 / 0.320 → 0.276 |
| four branches | 0.405 → 0.495 / 0.340 → 0.480 / 0.375 → 0.455 | 0.430 → 0.520 / 0.505 → 0.580 / 0.455 → 0.475 | 0.290 → 0.275 / 0.235 → 0.245 / 0.290 → 0.255 | 0.320 → 0.304 / 0.260 → 0.271 / 0.320 → 0.282 |

- Forgetting with the word cells alone is small: COCO things lose 0.03 on average over 6,000 moments (the lives lost
  0.10 in 10,000). So most of the life's loss is not this overwriting, or needs the rest of the life to show.
- Four branches change nothing that three seeds can see (after phase 2: 0.258 against 0.248 on COCO things). Every
  cell uses all four within the first phase: "less than half its usual drive" is true of too many pictures with
  these eyes, so the branches fill with ordinary variation and not with a new kind of look.
- Kept in the code (`Words(branches=…)`, off). Worth trying again only when a picture of a word is recognised
  reliably enough for "a new look" to mean something.

## The relay as a lateral-inhibition circuit (2026-11-17, `experiments/words_on_cortex.py`, arms lateral…)

Can the decorrelation be what a circuit with local synapses computes? Every neuron of the summary inhibited by the
others through synapses that hold their correlation (a Hebbian, local quantity); settled, the circuit passes on
level·(C + level)⁻¹ of its input. New one-layer cortex, full-size corrections, seed 0, held-out naming:

| relay | held-out naming |
|---|---|
| none | 0.388 |
| lateral inhibition, level 0.1 / 0.03 / 0.01 / 0.003 / 0.001 / 0.0001 of the largest variance | 0.447 / 0.473 / 0.490 / 0.487 / 0.452 / 0.362 |
| whitening computed from the statistics (in the lives now) | 0.533 |

The local form gives three quarters of the gain (0.39 → 0.49 of 0.53). So the relay need not be a batch computation:
correlation synapses can be learned online, in development and in life, and the settled circuit recomputed in sleep.
Not built yet; the lives run with the whitening.

## Fast mapping: absent, and an episodic store gives it (2026-11-17, `experiments/fastmap_probe.py`, `fastmap_life.py`)

The memory question in the form it has at this age: a name heard once with a kind of thing never named before (75
CIFAR kinds of later stages), asked at once and again after 20 ordinary naming moments. New cortex with relay,
word cells after 6,000 naming moments, seed 0 (seeds 1 and 2 running).

| | the same picture, at once | another of the kind, at once | the same, after 20 moments | another, after 20 |
|---|---|---|---|---|
| word cells, one lesson ("what is this?" – "fox") | 0.07–0.12 | 0.00 | 0.05–0.11 | 0.00 |
| word cells, three lessons on three pictures | 0.13–0.20 | 0.00 | 0.05–0.13 | 0.00 |
| word cells, named in an unasked comment ("look, a fox!"), once or three times | 0.00 | 0.00 | 0.00 | 0.00 |
| **a whole living brain** (world3b_s0 at 45,000 moments), one lesson, 40 kinds | 0.00 | 0.00 | 0.00 | 0.00 |
| word cells **with an episodic store**, one lesson | 0.91–0.92 | 0.00 | 0.88–0.89 | 0.00 |
| word cells with the store, named in an unasked comment | 0.00 | 0.00 | 0.00 | 0.00 |

(Ranges: corrections by the cell's own error and at full size; they do not differ here.)

- **The brain has no memory of a single moment.** What a two-year-old does after one hearing, it does not do after
  three, on the very picture it was shown. The living brain answers through the word cells (160 of 160 answers by
  recall), so the orchestrator's clusters do not hold a moment either. This is a missing piece, as Kris suspected.
- **An episodic store beside the word cells gives it** (`organic/episodes.py`): a moment the slow synapses did not
  expect is written whole at once, under what was seen (eyes closed: under what was heard); when the same situation
  returns its reply's units are driven; the oldest moment goes when the store is full (2,000 moments). The slow
  synapses learn exactly as before; first-words naming is not touched by it (0.430 → 0.463 with and without).
- **It is a memory of a moment, not of a kind**: another picture of the same kind brings nothing back. With one
  stored picture of each of 75 kinds, another picture of a kind is nearest to its own kind's 7% of the time (chance
  1.3%): these eyes do not make two foxes look alike enough. A child generalises a new name by shape at once; that
  waits for better eyes.
- **A name heard inside a comment is not given back as an answer**, with or without the store: what was stored is
  the whole comment, and the question asks for one word of it. Not looked into yet.
- Two numbers are set by hand in the store (how many moments, how steep the comparison). Not in a life yet.
