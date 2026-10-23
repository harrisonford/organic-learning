# The language pathway: what biology has that we do not (2026-10-22)

A review, not a change. The question: is something biologically basic missing from how the brain hears and speaks? This is not about how current language models work. Their techniques are only mentioned when they have a biological counterpart (BLT's entropy patches have one: infant statistical learning).

## What we have

```
bytes ─► EAR: next-byte statistics (1-4 bytes of context), units cut where
         surprise jumps, habituation, fixed random sparse codes per unit
      ─► "question code": fading echo of the last 8 units + a bag of all units
      ─► ORCHESTRATOR (with the eye's view) ─► one output neuron per whole answer
                                              ("word:on the left", "word:thank you")
```

| piece | biological counterpart | how faithful |
|---|---|---|
| next-byte statistics, cuts at surprise | infants find word boundaries where transitional probability drops (Saffran 1996) | good |
| habituation of frequent units | stimulus-specific adaptation in auditory cortex | good |
| sparse random codes for units and n-grams | combination-sensitive neurons | reasonable |
| echo of last 8 units + bag | auditory sensory memory (~2 s) | partial: fixed length, no rehearsal |
| one output neuron per whole answer | ? | **no counterpart**: speech is produced, not retrieved whole |
| learning answers by imitation (hearing what an adult says) | social / vocal learning | good in spirit |
| efference copy of own speech (`Ear.said`) | yes, speech motor efference | present but unused (answers are one shot) |

## What the world shows (full-world life, 330k moments)

Test items whose correct answer the brain can produce at all:

| kind | answer never heard *as an answer* | heard < 5 times (likely pruned) | answer is in the question / story |
|---|---|---|---|
| reason | 10% | 20% | **100%** |
| story | 0% | 0% | **52%** |
| chat | **39%** | **68%** | 3% |
| talk | 11% | 31% | 6% |
| photo kinds | 0-5% | 0-10% | 0-11% |

- **reason:** brain 0.05, against 0.44 for picking one of the offered options at random.
- **chat:** about 0.
- **story:** about 2x chance, the best text result.

The failures are where the pathway lacks a biological route, not where data is missing.

## Gaps, ranked

### 1. No route from hearing to saying (the dorsal stream)

Humans can repeat a word they have just heard for the first time. The route runs from auditory cortex (posterior superior temporal, area Spt) through the arcuate fasciculus to the motor speech areas (Broca, premotor). When it is damaged, people understand but cannot repeat (conduction aphasia, Hickok & Poeppel 2007). Infants build this route by babbling: they hear their own sounds and link heard patterns to the articulations that made them. Songbirds do the same.

Our brain can only say whole answers it has already heard *as answers*. A word it just heard in the question ("... field or hole?") cannot become what it says. This is the cleanest explanation of reason ≈ 0 and of half the story items.

*Biological shape of a fix:* speech output as motor units over the ear's own sound units. A babbling phase links each motor unit to the sound it makes (self-hearing). A copy route lets a unit echoing in auditory memory prime the motor unit that produces it. Which echoing unit gets said would then be learned like everything else, by imitation and reward. No pointer or attention mechanism: just the auditory-motor map that babbling builds.

### 2. Speech is retrieved whole, not composed

Every distinct reply ("you're welcome", "fine, thank you", "see you at the park") is a separate output neuron. Biology composes speech: words from a lexicon put into order by motor sequencing (Broca, premotor cortex, basal ganglia loops for sequence and chunking). Frequent phrases become chunks, but anything new is assembled.

That is why 68% of chat replies are effectively unsayable. It is also why the answer budget had to cut 20k answers.

*Shape of a fix:* produce in steps: one unit per tick, with the efference copy of what was just said (already there in `Ear.said`) feeding back as context. Stop when silence is produced. This also gives speech a duration, so fast and slow answers would differ naturally.

### 3. Hearing is instantaneous, and the auditory memory is fixed

The whole utterance arrives at once. Then the brain only sees an 8-unit fading echo plus an unordered bag. In biology:
- Speech unfolds over time, and comprehension is incremental: the brain predicts and binds as words arrive.
- A phonological loop (Baddeley) keeps a few seconds of speech alive by rehearsal.
- The hippocampus binds one-shot episodes ("mary went to the kitchen"). Those episodes are what a story question is answered from.

bAbI stories longer than about 8 units lose who-went-where. The repo has a hippocampus (`organic/hippocampus.py`, used by the earlier brain) that the loop brain does not use.

*Shape of a fix:* let listening take ticks, like looking does (one unit or a few per tick, so the orchestrator's state evolves while hearing). Plus fast Hebbian binding of co-occurring units (who + where) in a small episodic store that the question can cue.

### 4. Words are not grounded in what is seen

The ear's units meet the eye's features only inside orchestrator clusters made for whole moments. Children learn word meanings across situations: "dog" heard many times while dogs are visible strengthens a direct word-object link (cross-situational learning, Smith & Yu 2008). There is also a convergence zone for concepts (anterior temporal lobe "hub").

We have no direct, slowly accumulating link between a heard unit and visual features. That hurts presence ("is there a dog?"), which sits at the majority baseline.

*Shape of a fix:* a Hebbian association area between auditory units and visual / cortex features, updated whenever both are active. No labels: co-occurrence only.

### 5. No conversation context

Each moment is independent: the brain does not remember what was said a moment ago. The text world's "dialogue" format packs turns into one utterance, which is a workaround in the data, not memory. This is lower priority: it depends on 3.

## What not to add

- Tokenizers, embeddings or attention trained by backpropagation.
- Copy or pointer networks as such. The biological version of copying is gap 1: an auditory-motor map built by babbling.

## Recommendation

Do nothing to the architecture yet, as agreed: data first. When we do change it, the order would be:
1. **Babbling + auditory-motor copy route** (gap 1). It is the most basic biologically, it is testable on reason and story, and it is needed for 2.
2. **Composed speech, one unit per tick** (gap 2). It removes the need to prune answers.
3. **Listening over ticks + episodic binding** (gap 3).
4. **Cross-situational word-object links** (gap 4).

Each can be tried as a separate life against the same frozen world, like the cortex comparison.
