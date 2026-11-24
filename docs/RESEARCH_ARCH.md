# Architecture ideas from recent work (2023–2026): a survey, 2026-11-24

Asked by Kris on 2026-11-24 ("did we find more interesting sota research that we can take architecture ideas from
that are more recent?"). A survey agent searched and read abstracts or full texts; **none of these has been opened
by a person yet** (docs/paper/REFERENCES.md rule). Marks: [F] full text read by the agent, [A] abstract only.
What we already had (docs/RESEARCH_RECENT.md, RESEARCH_CORTEX.md, RESEARCH_GAZE.md) is not repeated.

## What it changes for us, in the order I would take them (rule 8: by how well established, not by score)

1. **Learning from what stays the same across successive looks** (a temporal term in the filters' rule).
   Halvagal & Zenke 2023 [F] (Nat. Neurosci. 26:1906; "LPL"): each neuron's rule has three local parts: a
   predictive term that pulls its responses to consecutive inputs together, a Hebbian term scaled by the inverse
   of its own variance (BCM-like), and anti-Hebbian decorrelation between neurons. Evidence: it reproduces the
   "swap" experiments in monkey IT (Li & DiCarlo) and the rate dependence of STDP. Without the predictive term the
   layer-local readout falls from 63.2% to 27.0% on STL-10, below pixels. Yu et al. 2024 [A] (arXiv 2411.01969):
   crops centred on a toddler's gaze, with a time-based objective, give view-invariant object codes, and the small
   central field is what matters; Schaumlöffel et al. 2026 [A] (ICLR 2026); Yu et al. 2025 [A] (cortical
   magnification helps). *For us:* our learner sees stills, so it had nothing that follows in time. The serial
   gaze changes that: the coarse view of a place and then the fovea's view of it, and successive fixations of
   one thing, are consecutive inputs of the same thing. The filters' rule gains the term "answer the next view
   as you answered this one". Our decorrelation steps are the rule's third part already. This is also the
   established route to complex-cell-like invariance (queue E 24, item 2).
2. **A blurred start** (the eye matures). Lu et al. 2025 [A] (arXiv 2507.03168): acuity, contrast sensitivity and
   colour maturing as in infants gives the strongest reliance on shape reported; Cai et al. 2025 [A] (arXiv
   2511.14440); Sheybani et al. 2023 [A]; Ligeralde et al. 2023 [A] (arXiv 2312.02791): filters first learned on
   retinal waves generalise better. *For us:* the cortex develops on sharp pictures from the first moment. A
   newborn's acuity is about a fortieth of an adult's. Development of the filters on waves, then on blurred and
   pale pictures that sharpen with age. Only a schedule on the retina; no new rule.
3. **Critical periods in sequence** (Delrocq, Wu, Bellec & Gerstner 2024/2026 [F], bioRxiv
   10.1101/2024.12.20.629674): plasticity windows open one area after the other, each after the one before it
   has closed; this helps local rules and hurts backprop. Their companion (arXiv 2605.18557 [A]): layerwise
   self-supervised rules learn hierarchical structure as data-efficiently as backprop; direct-feedback rules do
   not. *For us:* when the cortex gets a second stage, it develops after the first has settled (we did this by
   hand in `develop`; this is its ground), and the stage's rule is local to it.
4. **Surprise sets how much is learned** (neuromodulators as learning rates). Jordan & Keller 2023 [A] (eLife
   12:RP85111): the locus coeruleus broadcasts the unsigned size of prediction errors across cortex, and
   minutes of its stimulation give plasticity that otherwise takes days. Wilmes et al. 2025 [F] (eLife
   13:RP95127): an error divided by the learned variance of its context, (s − μ)/σ², with SST cells carrying the
   mean and PV cells the variance, both by local rules: a learning rate per context without a hand-set one.
   Coddington et al. 2023 [A] (Nature 614:294): dopamine adapts the rate of learning the policy rather than
   carrying a signed error. *For us:* the word cells had a step scaled by their own error (`plastic_by_error`)
   and the lives run with it off; the gating policy's steps are fixed numbers. One broadcast surprise that
   scales plasticity everywhere, and errors scaled by their own variance, would replace several hand-set rates.
5. **What comes back in sleep is chosen** (Yang et al. 2024 [A], Science 383:1478: awake ripples at reward tag
   what is replayed in sleep; Chang et al. 2025 [A], Nature 637:1161: recent and old memories replay in separate
   substates of sleep). *For us:* replay draws moments at random. Tagging by reward or surprise at the moment
   itself, and recent and old in separate blocks, are small changes to the store.
6. **A sparse expansion before the cells that associate** (mushroom body, dentate gyrus, cerebellar granule
   layer; Zou et al. 2026 [F], arXiv 2510.16877, on frozen pretrained features: engineering context; Rudelt et al.
   2025 [A], arXiv 2511.10261: these layers are themselves plastic; Chandra et al. 2025 [A], Nature 638:739: a
   fixed scaffold code as the store's index degrades gracefully instead of falling off a cliff). *For us:* a
   wide, sparse layer (the few most driven cells stay on) between the relay and the word cells and as the
   store's key. Well grounded in three brain structures; whether it adds anything beside our decorrelation is
   an open question, to be read on the code first.
7. **Surprise as a gain on features, not as error units** (Furutachi et al. 2024 [A], Nature 633:398: VIP cells
   inhibit the SST cells that gate pulvinar input to V1, amplifying the cells selective for the unexpected
   stimulus; Westerberg et al. 2024 [A]: sensory cortex shows little of the classical "global" prediction
   error). *For us:* a caution more than a part: do not build explicit error neurons into early vision.
Also noted, less ready: predictive coding that learns at every step and in both directions (Salvatori et al. 2024
[A]; Oliviers, Tang & Bogacz 2025 [F], arXiv 2505.23415: one bidirectional pair gives recognition and imagery);
temporal predictive coding (Millidge et al. 2024 [F], PLoS Comput. Biol. 20:e1011183); predicting the next
glimpse from the current one and the eye's own movement (Thorat et al. 2025 [F], arXiv 2511.12715: trained with
backprop on a frozen network, mixed evidence); layer roles (Nejad et al. 2025 [F], Nat. Commun. 16:6178, trained
with backprop); a Hebbian convolutional network with lateral inhibition and BCM at 75.2% on CIFAR-10 (Nimmo &
Mondragón 2025 [A]); sleep driven by noise with Hebbian changes (Kubo et al. 2025 [F]).
**Word learning.** Mutual exclusivity emerges without being built in, more so with more visual knowledge (Nortje
et al. 2024 [A], TACL): a free thing to observe in our learner (never a target). Objects tracked first and then
aligned with what is said add to child-view word learning (Silva et al. 2026 [A], arXiv 2606.12985); raw speech
and pictures at a real child's rate of naming give infant-like vocabulary growth (Khorrami & Räsänen 2025 [A]).

## The references, as the agent gave them (to be opened by hand before any is cited)
Halvagal & Zenke 2023, Nat Neurosci 26:1906–1915, doi:10.1038/s41593-023-01460-y · Delrocq, Wu, Bellec & Gerstner
2024/2026, bioRxiv doi:10.1101/2024.12.20.629674 · Delrocq et al. 2026, arXiv:2605.18557 · Nimmo & Mondragón 2025,
arXiv:2501.17266 · Salvatori et al. 2024, arXiv:2212.00720 · Oliviers, Tang & Bogacz 2025, arXiv:2505.23415 ·
Millidge et al. 2024, PLoS Comput Biol 20(4):e1011183 · Innocenti, Achour & Buckley 2025, arXiv:2505.13124 ·
Westerberg et al. 2024, bioRxiv doi:10.1101/2024.10.02.616378 · Furutachi et al. 2024, Nature 633:398–406 · Nejad
et al. 2025, Nat Commun 16:6178 · Wilmes et al. 2025, eLife 13:RP95127 · Hertäg, Wilmes & Clopath 2025, Nat Commun
16:3036 · Francioni et al. 2023, bioRxiv doi:10.1101/2023.11.03.565534 · Lv et al. 2025, arXiv:2501.09976 · Yu et
al. 2024, arXiv:2411.01969 · Schaumlöffel et al. 2026, ICLR 2026 · Sheybani et al. 2023, NeurIPS · Thorat et al.
2025, arXiv:2511.12715 · Yu et al. 2025, arXiv:2509.15751 · Rao 2024, Nat Neurosci 27:1221–1235 · Clay, Leadholm &
Hawkins 2024, arXiv:2412.18354 · Zou et al. 2026, arXiv:2510.16877 · Zou, Zang & Ji 2025, arXiv:2502.01427 ·
Rudelt et al. 2025, arXiv:2511.10261 · Chandra et al. 2025, Nature 638:739–751 · Bricken et al. 2023,
arXiv:2303.11934 · Jordan & Keller 2023, eLife 12:RP85111 · Coddington, Lindo & Dudman 2023, Nature 614:294–302 ·
Svahn et al. 2025, bioRxiv doi:10.1101/2025.11.12.688036 · Kubo, Delanois & Bazhenov 2025, arXiv:2508.14081 ·
Chang et al. 2025, Nature 637:1161–1169 · Yang et al. 2024, Science 383:1478–1483 · Zhou, Kahana & Schapiro,
eLife 13:RP99931 · Nortje et al. 2024, arXiv:2403.13922 · Khorrami & Räsänen 2025, arXiv:2406.05259 · Vong & Lake
2025, arXiv:2507.14749 · Silva et al. 2026, arXiv:2606.12985 · Zhuang, Fedorenko & Andreas 2024, arXiv:2403.14551 ·
Lu et al. 2025, arXiv:2507.03168 · Cai et al. 2025, arXiv:2511.14440 · Ligeralde et al. 2023, arXiv:2312.02791 ·
Cheon, Lee & Paik 2024, arXiv:2405.16731.
