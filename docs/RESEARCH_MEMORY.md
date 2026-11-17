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
