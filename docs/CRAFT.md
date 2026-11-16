# The craft: how this brain is worked on (kept up to date; started 2026-11-16)

Kris asked for the process to be documented, not only the results. This file is the method and the running
ledger of what was tried in each cycle and why. The detailed numbers live in the files it points to.

## The two questions, in this order

1. **Is it how living brains and beings work?** A mechanism is proposed with its biology first (who showed it, in
   what animal or child, what the rule is). A performance trick with no counterpart in life is not built, however
   well it would score. If the biology is contested, that is said.
2. **Is the answer more or better experience instead?** Before changing the architecture: is there enough to see
   and hear, is it of the quality a child gets (things in view while they are talked about, replies to what the
   child itself said, things seen whole and many times), has it lived long enough, are there enough layers for
   what is asked? Composite moments, seeing and hearing together, are the normal case in life and should be here.

Only then: does it help, measured.

## Working rules

- **Measure the ceiling before building.** For any pathway, first ask what a simple reference learner (a count, a
  nearest neighbour, a linear or kernel readout fitted in a batch) gets from the same input. References are not
  brains; they locate the loss. Three times this cycle the reference said "the loss is not where the plan
  assumed": the last-word parrot, the unanswerable photo questions, the ear's cuts.
- **Offline first, three seeds, then a life.** A probe on cached data takes minutes; a life takes a day. A claim
  needs three seeds (three orders of moments) and 300-item tests. A single seed is called a single seed.
- **One change per life**, so that a life answers one question. When two things changed, the confound is written
  down.
- **Negative results are kept**, with the code, and with what was learned from them. A design that fails is
  usually the fastest way to the next one (the lexicon ear → the carving ear → the edge ear).
- **Hand-set numbers are debts.** Each is named where it stands, and removed when the brain can measure the
  quantity itself (the copy route's 0.75 recency; the ear's two thresholds are still owed).
- **Health before any report** (`experiments/health.py`), long jobs under `systemd-run`, probes capped in memory so
  that a life is never the process that dies.
- **Everything is written twice**: here in `docs/` in full, and one sentence per decision or finding in memoize.

## Ledger

### Cycle of 2026-11-10 to 11-16 (after "L3 only repeats the last word")

| what | organic basis | data question | result | where |
|---|---|---|---|---|
| word cells in place of the hub (unit cells, two dendrites, company links) | dendritic prediction (Urbanczik & Senn 2014); fast and slow learning systems (McClelland et al. 1995) | – | pictures 0.29 → 0.44 offline; lives pass first words; stop at two words | AREAS.md |
| the copy route's hand-set recency removed; novelty from the ear's own counts, the heard words competing | repetition suppression; echoing as the default when meaning is silent (dual stream) | echo games reworded by the adults so the word is not always last | echo with the word anywhere learned; last word said in 6–17% of replies | CONVERSATION.md §6 |
| one stream instead of three routes | one motor plan fed by inner and outer speech | – | negative: the word cells always have something to say | CONVERSATION.md §6 |
| the parrot beside every score, and as the gate | – (a yardstick) | the lessons' own last-word habit is real in speech to infants and stays | a stage is no longer passed on what a parrot answers | CONVERSATION.md §6 |
| heard words bringing back what was seen (feedback synapses, bursts) | cell assemblies (Pulvermüller); labels shape perception (Lupyan) | – | no effect; capped by the eyes and by the ear's units | CONVERSATION.md §6 |
| sound of what was heard as an input to the word cells | combination-sensitive auditory neurons | – | +0.05 same answer across wordings, little else; not adopted | CONVERSATION.md §6 |
| a lexicon ear (cheapest explanation by known units) | known-word segmentation (Brent & Cartwright 1996) | – | negative: glues collocations, naming 0.21 | EAR_CARVING.md |
| words heard alone carve longer speech | free forms; isolated words (Brent & Siskind 2001); possible-word constraint (Norris et al. 1997) | adults' one-word answers are the isolated words | names consistent across settings; text unchanged; kept as the teacher | EAR_CARVING.md |
| the ear learns what word edges sound like | infants learn language-specific edge cues from known words (Jusczyk et al. 1999; Mattys & Jusczyk 2001) | – | offline: parrot-proof text +0.10 in 3 seeds. In lives: text +0.08, pictures −0.05, first words passed later | EAR_CARVING.md |
| a word is the same word whatever mark follows it | pauses and marks learned by the ear itself | – | offline: text +0.02 more, pictures unchanged | EAR_CARVING.md |
| word cells with their own firing level | a neuron's threshold between its own two levels of input | – | more often right when they speak (pictures 0.35 → 0.48), silent on half of new-word echoes instead of a quarter, but fewer answers overall | CONVERSATION.md §7 |
| replaying the day in sleep | hippocampal replay (Wilson & McNaughton 1994) | it is "more training" on the same experience | pictures +0.02 in 3 seeds, text nothing; not adopted | CONVERSATION.md §7 |
| leisure talk: a parent answers what the child said, nothing expected | contingent replies (Goldstein & Schwade 2008); recasts (Brown & Bellugi 1964) | a new kind of experience, not a mechanism | built; not yet measured in a life | LEISURE.md |
| reference ceilings for the eyes | – (a measurement) | naming rises with pictures seen, slowly beyond 8,000 with this cortex | the code holds 0.55; the two-words photo questions are not answerable from it | VISION.md |
| a bigger picture book (95,000 things cut from COCO photographs) | a child sees a dog thousands of times | yes: this is the data answer | built (world 3) | VISION.md |
| things in view when they are talked about (world 3) | joint attention; seeing while hearing is the normal case | yes | built; not yet measured | CONVERSATION.md §7 |
| pruning in sleep | overproduce and remove (Huttenlocher 1979); least used goes (Navlakha et al. 2015); sleep scales down sparing the most used (de Vivo et al. 2017) | – | literature read; first probe running | PRUNING.md |
| the cortex taught by the words; more to see for the cortex | feedback-gated plasticity (Roelfsema & van Ooyen 2005); perceptual learning by category (Sigala & Logothetis 2002) | the cortex developed on 4,000 pictures per layer; its top neurons see 17 of 64 pixels | running | VISION.md |

### What the cycle taught about the method

- The tests were paying for the wrong thing twice (the last word; photo questions nobody could answer). Checking
  the yardstick comes before reading it.
- "The limit is X" was wrong whenever it was inferred from a score and right whenever it was measured with a
  reference on the same input.
- A change that helps offline can lose in a life (the new ear and picture naming). The life is the only test that
  counts, and it is slow; so the offline probe must look like the life (the echo stage first, the same budgets).
