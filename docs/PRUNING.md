# Memory-budget pruning (a compromise, to be revisited)

**Decision (2026-10-22):** long lives run under a hard memory budget. When the brain reaches it, the least valuable clusters and the least heard answers are removed. This is **not** a learning rule we believe in. It keeps experiments going on a 14 GB machine, and it is written down so it can be undone or replaced later.

## Why

The full world has 1.17M items and **24,593 distinct answers**.
- Every answer the brain hears becomes an output neuron.
- Every cluster has 8 neurons with a dense output weight to every answer.
- So memory grows as clusters x answers.
- Storage doubled whenever it filled: 6.9 -> 11.9 GB in one step at 200k moments.

Per cluster (input dim 1,760, 4,096 answers): ~190 KB, of which 70% is output weights.

## What it does (`Cortex.set_budget`, `prune`, `prune_outputs`, `fit_capacity`)

- **Cluster budget** (`MAX_CLUSTERS`, 24,000 in `full_s0`):
  - When it is reached, the lowest-value 10% of clusters are removed.
  - Value = usage / (1 + idle / 30,000), where usage is how many learning events a cluster took part in, and idle is the time since it last took part.
  - Clusters younger than 5,000 learning events go last, only if there are not enough older ones (the budget is hard).
- **Answer budget** (`MAX_OUTPUTS`, 4,096):
  - When a new answer is heard and the budget is full, the 10% least valuable answers are forgotten. Their output synapses are removed from every cluster.
  - Value = times heard / (1 + time since last heard / 30,000).
  - Only `word:` outputs are pruned; eye and gain commands never are.
  - 4,096 roughly covers every answer heard at least 5 times in the world (3,778).
- **Storage** is allocated up to the budget (+64 slack) and never doubles past it. `fit_capacity` returns memory when a budget is imposed on an existing brain.
- Pruning and sleep now run **at the end of a moment**, after every use of that moment's cluster indices.
  - Before, sleep (every 2,000 moments) ran just before patience tuning and command credit, which then went to shifted indices.
  - That small bug is fixed for all lives.

## What it costs, honestly

- **Rare things are forgotten.** An answer heard a few times (a rare VQA answer, a name in a story) loses its output neuron. If it is heard again it is relearned from scratch.
- **Old, rarely used memories die**, even if they would be right the next time their situation comes. This is "use it or lose it", which brains do, but here the trigger is the budget, not anything the brain experiences.
- **Results are not comparable** with unbudgeted lives once pruning starts (the `prune_log` in each record says when it did).

## How to undo or improve it later

- With more memory: drop `MAX_CLUSTERS` / `MAX_OUTPUTS` (default off); nothing else changes.
- Store output weights sparsely. Most clusters only ever point at a few answers, so behavior would be identical and the memory cost would mostly vanish. This is the first thing to try.
- Replace the budget trigger with something the brain experiences: synaptic homeostasis during sleep, or metabolic cost per cluster.
