# Pruning in sleep: what biology and machine learning say (2026-11-16)

Kris: *"there was some research that pruned models heavily, reducing size and keeping behaviour; check if
something there can be done to let us push levels without inflating the architecture, akin to sleeping
consolidation."* A literature pass (a research agent; it read full text for Navlakha, SET, Tadros 2022, Scholl,
Frank, OBD, DeepR, RigL, lottery ticket and Wanda; the rest from abstracts) and the plan that follows from it.

## Biology

| finding | the rule it suggests | numbers |
|---|---|---|
| Huttenlocher 1979; Huttenlocher & Dabholkar 1997; Petanjek et al. 2011 | make too many synapses, then remove; sensory areas finish first, associative ones last | frontal synapse density peaks at 1–2 years about 50% above adult; childhood prefrontal spine density 2–3× adult; removal goes on into the third decade |
| Chechik, Meilijson & Ruppin 1998 | remove the weakest | with Hebbian synapses under an energy limit, overgrow-then-prune maximises memory (theory) |
| Navlakha, Barth & Bar-Joseph 2015 | remove the least used, fast at first and then ever slower | mouse barrel cortex: about 40% pruned in days 19–29, 11% in days 29–39; a decreasing rate is 20–30% more efficient than a constant one |
| Tononi & Cirelli 2003, 2006, 2014 (synaptic homeostasis) | waking strengthens on balance; sleep scales back; synapses used again and again are protected, the rest are weakened and in the end removed | receptor levels 30–40% higher after waking |
| de Vivo et al. 2017; Diering et al. 2017 | scaling is multiplicative and spares the large synapses | synapse area about 18% smaller after sleep; about 80% of synapses scale, the large stable ones do not |
| Maret 2011; Li et al. 2017; Yang et al. 2014 | sleep removes on balance; REM removes newly formed spines and strengthens a few of them, which frees room for later | (no numbers in the abstracts) |
| Stevens 2007; Schafer 2012 | synapses are tagged (complement) and eaten by microglia, depending on activity | (no percentages in the abstracts) |

Against a simple "sleep shrinks everything": Frank 2012, Chauvette et al. 2012, Durkin & Aton 2016 (responses
also get stronger over sleep). Sleep reorganises selectively. Three criteria overlap in the biology: the weakest,
the least used, the newest not yet consolidated.

## Machine learning

| work | what is removed | how much | local? | regrowth |
|---|---|---|---|---|
| LeCun, Denker & Solla 1990 | lowest second-derivative saliency | 60%, error unchanged after retraining | no | no |
| Han et al. 2015 | smallest weights, then retrain | 9–13× smaller, no loss | the criterion yes, the retraining no | no |
| Frankle & Carbin 2019 (lottery ticket) | smallest weights, 20% a round | 10–20% of the size left, accuracy slightly *higher* | criterion yes | no |
| Mocanu et al. 2018 (SET) | each epoch the 30% closest to zero | 1–2% of the weights; beat the dense network | yes | random |
| Bellec et al. 2018 (DeepR) | a connection sleeps when its weight crosses zero | 96.2% on MNIST at 1.3% connectivity | rewiring yes | random |
| Sun et al. 2024 (Wanda) | weight × how active its input is, ranked per receiving unit | 50% with no retraining (plain magnitude fails there); collapses at 80% | yes | no |
| Scholl, Rule & Hennig 2021 | pre × post coincidence (a local estimate of importance) | error near baseline until >10× fewer weights, with retraining; removes whole units | yes | no |
| Tadros et al. 2022 | sleep on noise: strengthen if pre and post active, weaken if post active and pre silent | no pruning measured; recovers forgotten tasks | yes | – |

Pruning improves generalisation at moderate sparsity (LeCun, Frankle, Mocanu, Averbeck 2022), never at extreme
sparsity, and it costs some ability to learn new things later (Robinson 2022; Averbeck 2022). The heavy numbers
(9–50×) all rely on retraining by gradient; under local rules the evidence is small-scale.

## Organic check, and what fits this brain

Pruning in sleep is how brains are built (overproduce, then remove what was not used), so it passes the first
test. What the brain here already has: clusters consolidate and retire in sleep (use it or lose it), word cells and
cues are forgotten by use when their room is full. What it does not have: its synapses are never removed. Every
word cell keeps 7,168 visual synapses and one per cue, used or not, and a level's size on disk is mostly synapses.

Candidates, in the order of their support:

1. **Per dendrite, remove the synapses lowest in weight × use** (use-it-or-lose-it, Navlakha; Wanda's criterion).
   The synapses' use counters already exist.
2. **Scale down in sleep, sparing the most used; what falls to nothing is gone** (de Vivo; Tononi & Cirelli).
3. **Overgrow, then prune fast at first and ever slower** (Huttenlocher; Navlakha).
4. **A fixed room with turnover**: removed synapses free room for new ones on the next day (Li et al. 2017; SET).

What will be measured first, offline: how many of a word cell's synapses can go, by rule 1 against plain smallest
weight, before naming changes; whether naming of pictures never seen *improves* at moderate pruning; and whether
learning afterwards is slower. Then the same for the orchestrator's clusters, which are most of a level's size.
