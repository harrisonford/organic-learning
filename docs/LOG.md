# Running log (scale-up phase)

Plan agreed 2026-10-17: free models act as adults in the brain's world
(write and check questions, grade, compare), never inside the brain; half of
the account's limits (<=10 req/min, <=500 req/day); Jev capped at $1.50;
everything detached and resumable; long lives.

## 2026-10-17

- **Adults** (`organic/adults.py`): rate-limited, cached OpenRouter client;
  per-purpose daily shares: photo annotation 340, text world 100, reserve 60.
  Nemotron 3.5 Super VL writes, Dots 3 Note checks (both free, see images).
  Inkling Small free is gated to agent apps (403), so it is not used. Jev is
  `typesafe/jev-router`: a router running on TypeSafe's system-one model that
  can pick paid models, so it is capped.
- **Pilot:** 4 COCO photos at the eye's resolution, 5 questions each: the two
  adults agreed on 19/20 answers.
- **Daemons running:** COCO photo annotation (~700 photos/day), text world
  (spoken Q&A for the ear, 50 pairs per request).
- **World** (`organic/world.py`): COCO val2017 photos at 64x64 with questions
  from human annotations (presence, counting, where, biggest thing), adult
  questions (agreed items only), reading (words drawn on photos), eyes-closed
  talk, and CIFAR-100 once downloaded. Stable photo-level test split;
  kind-balanced sampling; hidden deadlines per kind.
- **Sleep consolidation** (`Cortex.consolidate`): merges a new cluster into an
  existing one only if they stand for the same situation *and* expect the same
  outcome; retires long-idle, barely used clusters. Finding: new clusters'
  nearest neighbours have similar identities (median 0.90) but different
  expected outcomes (output similarity 0 in >90% of pairs), so growth is not
  duplication; merging is rare by design and retirement does most of the work.
- **Runs:** sleep-cleanup lives on the Yarbus task (design 2 + contrast reflex
  + adaptation, 80k moments, 2 seeds); photo-world lives (300k moments,
  2 seeds); counterpart benchmark scheduled for tomorrow's reserve.

### Photo world, first checkpoints (age 10,000-20,000 of 300,000)

Baseline = always giving the most common training answer for that kind.

| kind | brain | baseline | answer tick |
|---|---|---|---|
| reading (48 words) | 0.18 -> 0.39 | 0.02 | ~10-11 |
| presence | 0.61-0.66 | 0.61 | ~1.1 |
| where is the person | 0.38-0.52 | 0.49 | 3-4 |
| counting | 0.49-0.51 | 0.56 | 3-4 |
| identity ("biggest thing", 83 answers) | 0.20-0.23 | 0.32 | 7-8 |

Reading is clearly learned early; presence is answered almost instantly and
reading / identity late, matching their hidden deadlines. Other kinds are at or
below the majority baseline so far. Color, gist, activity and talk have too
few test items yet (3-5) to judge; they grow as the adults write more.

## 2026-10-18

### Sleep cleanup (design 2 + contrast reflex + adaptation + sleep, age 80,000)

| | color | shape | gaze -> asked side (shape) | clusters |
|---|---|---|---|---|
| without cleanup (seeds 0/1) | 0.78 / 0.83 | 0.68 / 0.62 | 0.82 / 0.90 | 17,930 / 16,266 |
| with cleanup, seed 1 (age 80k) | 0.78 | 0.61-0.64 | 0.88-0.89 | 15,202 |
| with cleanup, seed 0 (age 74k) | 0.78 | 0.54-0.68 | 0.62-0.86 | 17,634 |

Cleanup trims growth modestly (seed 1: -7%) without hurting accuracy; it
cannot stop growth, because new clusters are distinct memories, not copies.

### Counterpart benchmark (80 held-out items, 4 per request)

| model | presence | count | identity | where | reading | s / request |
|---|---|---|---|---|---|---|
| Nemotron 3.5 Super VL (free) | 0.94 | 0.69 | 0.31 | 0.00* | 1.00 | 17.1 |
| Dots 3 Note (free) | 0.94 | 0.06 | 0.31 | 0.00* | 1.00 | 16.2 |
| Jev router (paid flash models) | 0.81 | 0.38 | 0.25 | 0.00* | 1.00 | 5.1 |

*"where" scores 0 only because the models answer "left" and the expected
answer is "on the left" (exact matching); not a real failure. Jev routed to
DeepSeek / Gemini / GLM flash models and cost $0.035 for 20 requests.

### Photo world at age 100,000: not learned

| kind | seed 0 / 1 | baseline |
|---|---|---|
| presence | 0.51 / 0.52 | 0.61 |
| counting | 0.46 / 0.49 | 0.56 |
| identity | 0.21 / 0.19 | 0.32 |
| where | 0.47 / 0.42 | 0.49 |
| reading | 0.24 / 0.39 | 0.02 |

Diagnosis (nearest-neighbour probe on the eye's output): for real photos the
parts-code eye carries no usable information for presence, identity or where
(probe at or below baseline); only reading (0.32). The eye organ, built for
coloured shapes on black, is the bottleneck: the brain cannot learn what the
eye does not deliver.

Same probe on the retina eye with its developed V1 (stored as `retina-eye-v1`):

| question | baseline | 1 fixation | 3 fixations |
|---|---|---|---|
| is there a person? | 0.57 | 0.57 | 0.57 |
| reading | 0.02 | 0.36 | 0.36 |
| biggest thing | 0.26 | 0.14 | 0.14 |
| where is the person | 0.44 | 0.35 | 0.34 |

No better. For real photos, no raw retina-like code (with or without V1)
makes these questions separable by similarity. In animals that job belongs to
a deep, wide visual cortex grown during development (V1 -> V2 -> V4 -> IT; the
fly's 77,000-neuron optic lobes); this brain has nothing between the organ and
the orchestrator. Proposed: let such a hierarchy self-organize on natural
photos with the same growth and Hebbian rules.

## 2026-10-19: visual cortex

`organic/visual_cortex.py`: three retinotopic layers (V1 -> V2 -> V4/IT-like)
learning with the same competitive Hebbian rule as V1 and the parts bank, on
contrast-normalized windows, sparse and max-pooled; developed layer by layer on
unlabeled training photos (`experiments/develop_cortex.py`).

Nearest-neighbour probe on held-out photos (baseline = most common answer):

| question | baseline | eye alone | small cortex L1 / L2 / L3 | big cortex L1 / L2 / L3 |
|---|---|---|---|---|
| CIFAR-100 identity | 0.01 | 0.07 | 0.08 / 0.09 / 0.10 | 0.11 / 0.10 / 0.08 |
| is there a person? | 0.57 | 0.55-0.58 | 0.60 / 0.55 / 0.58 | 0.60 / 0.54 / 0.59 |
| biggest thing | 0.32 | 0.12-0.14 | 0.25 / 0.27 / 0.28 | 0.22 / 0.24 / 0.26 |
| where is the person | 0.40 | 0.40-0.45 | 0.27 / 0.34 / 0.35 | 0.30 / 0.32 / 0.39 |
| reading | 0.04 | 0.32 | 0.09 / 0.07 / 0.08 | 0.07 / 0.05 / 0.07 |

(small: 64/128/256 kinds, 6,000 samples per layer; big: 128/256/512 kinds, 30,000)

- The cortex adds real but modest information about *what* is in a photo
  (identity 0.10 vs 0.07 for the eye alone, 10x chance), and in the small
  cortex deeper layers do better.
- Size of a fixed cortex is not the lever (big = small).
- Pooling loses letter detail, so reading stays with the eye's fovea; the
  orchestrator receives both.

Wired into the loop brain: layers 2 and 3, pooled over the whole view (384
numbers), arrive at ticks 2 and 3. Very long lives started: seed 0 to
1,000,000 moments, seed 1 to 500,000, resumable, memory logged.

## 2026-10-21: long lives finished; one fixed test set for all brains

Lives finished: cortex seed 0 at 1,000,000 moments (~43 h), seed 1 at 500,000.
Size at the end: seed 0 has 57,038 clusters (456,304 neurons), ~1.19 billion
numbers (identities 100M, input synapses 803M, output synapses 284M); seed 1
~0.76 billion. Organs + cortex: ~0.37M. Only a handful of clusters respond per
tick, and most numbers are stored situation prototypes, so the count is not
comparable to a 1B-parameter language model's shared features.

All saved brains on one fixed held-out set (`experiments/compare_brains.py`,
up to 200 per kind; baseline = most common training answer):

| kind | baseline | no cortex, 300k (2 seeds) | cortex, 1M / 500k |
|---|---|---|---|
| talk | 0.09 | 0.27-0.29 | 0.55 |
| color | 0.13 | 0.36 | 0.41-0.42 |
| identity | 0.04 | 0.09 | 0.10-0.14 |
| gist | 0.92 | 0.65-0.67 | 0.79-0.82 |
| counting | 0.41 | 0.31-0.34 (tick ~6.6) | 0.35-0.39 (tick ~3.5) |
| where | 0.41 | 0.62-0.65 | 0.57-0.59 |
| reading | 0.01 | 0.23-0.36 | 0.20-0.21 |
| activity | 0.19 | 0.29-0.31 | 0.27-0.29 |
| presence | 0.90 | 0.80-0.81 | 0.79-0.82 |

Cortex brains do better on "what" (color, identity, gist) and count faster;
slightly worse on reading and where (pooling blurs position and letters). The
talk gain is mostly data: spoken questions grew while these lives ran. The two
groups also differ in life length and world size, so this is not a clean
cortex-only comparison. Presence, gist and counting stay below the majority
baseline. Most progress happens in the first 50-100k moments; later growth
(to ~1.2 billion numbers) mostly adds memory, not skill.

## 2026-10-21 (evening): richer world, controlled cortex comparison

**World enriched** (details in [WORLD.md](WORLD.md)):
- VQA v2: 21k human Q&A on our 5,000 photos.
- COCO train2017: 118k photos as 64x64 thumbnails, with 1.08M annotation and VQA items.
- Text kinds with their own hidden deadlines: chat (2 ticks), talk (3), reason (none), story (none).
- New text sources: DailyDialog (8.9k replies), CommonsenseQA + ARC-Easy + WinoGrande (45k), bAbI (8k stories).
- Text daemon v2: four formats, judge check instead of exact match, de-duplication, re-judging of old rejected pairs.
- Budget moved to text: 240 text / 200 photos / 60 reserve per day.

The full world is now 1.17M lived items (was ~150k).

**Controlled comparison:** fresh lives in one frozen world (`data/world_ctl`), identical except for the visual cortex, two seeds each, all stopped at 100k moments, 60 test items per kind.

| kind | cortex (s0, s1) | no cortex (s0, s1) |
|---|---|---|
| identity | **0.21** (0.22, 0.20) | 0.14 (0.15, 0.13) |
| gist | **0.80** (0.78, 0.82) | 0.70 (0.77, 0.63) |
| story | 0.44 (0.48, 0.40) | 0.39 (0.43, 0.35) |
| count | 0.38 | 0.35 |
| talk | 0.35 | 0.34 |
| yesno | 0.39 | 0.38 |
| activity | 0.25 | 0.23 |
| color | 0.28 | 0.32 |
| text (reading) | 0.13 (0.12, 0.15) | 0.24 (0.08, 0.40) |
| presence | 0.78 | **0.88** (0.87, 0.90) |
| where | 0.49 | **0.60** (0.60, 0.60) |
| ask | 0.02 | 0.05 |
| chat / reason | 0.00 | 0.00 (only 20 / 43 items in this frozen world) |

Standard error per cell is about 0.06. Results:
- Consistent across seeds: the cortex helps identity (+0.07) and gist (+0.10), and hurts where (-0.11) and presence (-0.10).
- The cortex code is pooled over the whole view, so "where" information is lost there. That fits the earlier long-life result, now without the life-length confound.
- Reading is noisy across seeds.

**Growth:** in the richer world a life has 28-33k clusters and 6.6-9.4 GB at 100k moments. Earlier it was 17k clusters at 100k. More kinds of task make the structure grow faster.

Lives had to run one at a time, because two together exhausted the 14 GB machine. `world_life.py` now stops cleanly when `MAX_GB` is exceeded.

**Running:** `out/full_s0` is one life in the full 1.17M-item world (cortex, seed 0, world refreshed every 10k), capped at 11 GB.

## 2026-10-22: a replay bug in long lives

The first full-world life (`out/full_s0_replayed.*`) reached reward 0.91 by 170k moments. Then reward fell to 0.62 within 30k moments, while clusters jumped from 18k to 25k and memory from 6.9 to 11.9 GB, which stopped the life.

**Cause:** a bug in `world_life.py`. Every 10k moments the world is rebuilt to pick up new adult questions. The rebuild also re-created the moment generator with the same seed. So whenever the data files had not changed (the daemons sleep once their daily budget is spent), the brain relived **the same 10,000 moments** again and again. The high reward was memorizing a loop. The drop came at midnight, when the text daemon resumed and the sequence changed.

**Affected:** every refreshing life:
- `world_s0`/`world_s1` (300k);
- `cortex_s0`/`cortex_s1` (1M / 500k);
- `full_s0_replayed`.

Conclusions drawn from them are unreliable, especially "most learning happens in the first 50-100k moments" and the long-life plateaus. **Not affected:** the controlled comparison (`ctl_*`), which used a frozen world with no rebuilds.

**Fix:** the moment generator is carried across rebuilds. `full_s0` was restarted from scratch.

Also: free-model replies sometimes carry an error body without `choices`. These are now retried instead of failing the batch (31 failures on 10-21/22). The text daemon made only ~15 successful batches in its first day (~260 new items), because of those failures. Judge acceptance by format: fact 114/129, story 60/60, chat 35/39, logic 29/29.

## 2026-10-22: memory-budget pruning

Long lives now run under a hard budget: 24,000 clusters and 4,096 answers. When a budget is reached, the least used and least heard 10% are pruned. Full rationale, costs and how to undo it: [PRUNING.md](PRUNING.md).

Also fixed: sleep ran before this moment's patience and command credit, which then went to shifted cluster indices. Sleep and pruning now run at the end of a moment.

`full_s0` was restarted with the budget. The previous unbudgeted run is kept as `full_s0_nobudget`; it had reached 10k moments.

## 2026-10-22 (night): richer adults, services, language review

- **Text daemon v3:**
  - 11 formats: fact, why, define, wordplay, chat, dialogue, logic, analogy, order, story, mind.
  - ~90 knowledge domains, 23 everyday situations, 15 people asking.
  - All four free models take turns writing; a different one judges.
  - A batch whose judging failed is kept and judged later (no wasted writer requests).
  - The adults client reports what error replies say.
- **Photo daemon v2:** each batch focuses on two of 14 angles (people, scene, materials, humour and danger, what happens next, everyday sense, animals, food...). The other model judges the writer's answers (as in text). After the 5,000 val photos it continues into the 118k train thumbnails.
- **Reboot:** the machine rebooted on 10-22 (~22:07); all processes ended. The full life resumed from its 330k save. A resumed life now draws moments from an rng seeded by its age, so it does not relive its first moments.
- **Systemd user services** (`experiments/systemd/`, installed in `~/.config/systemd/user`; lingering is on): `organic-text`, `organic-photo` (restart always) and `organic-life` (`experiments/life_service.sh`, resumes the current life from its last save). Reboots no longer stop the project.
- **Language pathway review:** [LANGUAGE_BIOLOGY.md](LANGUAGE_BIOLOGY.md).
  - The answer is in the question for 100% of reason items and 52% of story items.
  - 39% of chat replies were never heard as an answer.
  - The missing pieces are biological routes, not data: hearing-to-saying (dorsal stream, built by babbling), composed speech, listening over time with episodic binding, and cross-situational word grounding.
  - No architecture change yet.

## 2026-10-23: monitoring

`experiments/monitor.py` runs every 6 hours (systemd timer `organic-monitor`). It writes `out/monitor/report.md` and appends to `out/monitor/history.jsonl`. The report covers:
- world size per kind, with the majority-answer baseline;
- text formats;
- the daemons' daily yield;
- the life per kind: accuracy vs baseline, trend, ticks;
- the ear: units, mean unit length, how probe sentences are cut.

Life records now also carry, per kind:
- `modal`: the share of the most common reply (collapse onto one answer);
- `copied`: whether the reply is a word heard in the question;
- `producible`: whether the right answer exists as an output neuron right now.

The report raises flags that point at the gaps in [LANGUAGE_BIOLOGY.md](LANGUAGE_BIOLOGY.md).

Brain saves are now atomic (written to a temporary file, then renamed), so a restart during a save cannot corrupt the life.

## 2026-10-23 (afternoon): monitor findings at 610k moments

- **Word units have emerged in the ear without being taught.** "what | color is | the | ball?" and "mary | went | to | the | kitchen. | where is | mary?". The ear has 108k units, mean 5 bytes. Common pairs ("color is", "where is", "there a") are heard as one unit.
- **Collapse onto common answers:**
  - count: 53% "zero";
  - identity: 53% "person";
  - yesno: 55% "yes";
  - gist: 78% "yes";
  - "ask" (open questions): 45% "yes".
  The brain falls back on frequent replies.
- **Copying:**
  - story: 38% of replies are a word heard in the story, and accuracy is 0.38 (+0.20 over majority, the best kind).
  - reason: the answer is in the question for 99% of items, but only 18% of replies repeat a heard word, and accuracy is 0.04.
  Story's "copies" are words it also learned as answers ("kitchen"). This is not a hearing-to-saying route.
- **Producible:** only 50% of chat answers exist as output neurons (whole-answer output). It is 70% for talk and open questions, 80% for reason.
- **Photo kinds keep drifting down** under the budget: where 0.23 (-0.11 over the last 6 records), color 0.12, count 0.26. Reward is 0.19.
- **Text daemon:** xing4.0 dropped (0 usable batches in a day). The day had 31 good text batches and 27 failures.

## 2026-10-23: how the ear cuts speech at 670k moments (`experiments/ear_breaks.py`)

Units come only from the ear's own next-byte surprise (no tokenizer, no spaces rule).

#### photo questions
`what  | color is  | the  | bus?`
`how many  | people are  | in the  | picture | ?`
`is  | the  | man  | holding |   | an umbrella | ?`
`what  | is  | the  | woman  | doing?`

#### everyday chat
`thank you |   | very  | much!`
`good |   | morning | ,  | how a | re  | you | ?`
`see you |   | tomorrow |   | at  | school | .`
`i'm  | sorry | ,  | i  | didn't  | hear  | you.`

#### stories (bAbI)
`Mary  | moved to  | the  | bathroom | .⏎ | John  | went  | to  | the  | hallway | .⏎ | Wh | ere is  | Mary?`
`Daniel  | picked  | up  | the  | football | .⏎ | Daniel  | went  | to  | the  | kitchen.⏎ | Wh | ere is  | the  | football | ?`

#### reasoning
`the  | trophy  | doesn't  | fit in the  | suitcase |  because  | the  | ...  | is  | too  | big.  | trophy  | or  | suitcase | ?`
`what  | is  | 7 pl | us  | 5?`
`hot  | is  | to cold |   | as  | up is  | to what?`

#### never heard like this
`the  | purple e | lephant |   | sang |  qu | iet | ly  | un | der |  the  | bridge | .`
`astronauts |  eat |   | de | hydrate | d  | strawberries.`
`my  | gra | ndmother's  | bicycle  | is  | old | er than  | me.`

#### made-up words
`the  | blo | rp | y  | zi | ndl | e flomped |   | the  | gax | .`
`glim | merwi | ck  | and  | snor | fl | e went  | to  | thar | n.`

#### other languages / symbols
`¿ | dó | nde  | está |  la |  biblio | te | ca | ?`
`je  | ne sais pas.`
`3.14159 is  | pi, |  2+2 | =4`
`🙂 hello  | 🙂`

#### Word boundaries
66% of the ear's breaks fall at a word start; 77% of word starts are breaks.

#### Most common multi-word chunks
`there a ` (62,325), `how many ` (48,106), `where is ` (45,665), `color is ` (37,716), `people are ` (32,521), `in the ` (23,505), `sport is ` (21,339), `on the ` (14,204), `travelled to ` (13,306), `back to ` (13,026), `journeyed to ` (12,978), `moved to ` (11,490), `ere is ` (9,862), `do you` (8,044), `color are ` (7,824), `to the ` (6,658), `north of ` (5,614), `south of ` (5,596), `east of ` (5,589), `garden is ` (4,804), `How do ` (4,456), `where do` (4,370), `ved to ` (3,766), `are the ` (3,751), `bigger than ` (3,583)

#### Most common word pieces (no space inside)
`you` (24,094), `pla` (18,374), `Wh` (17,988), `out` (6,028), `ha` (5,979), `like` (5,739), `right` (5,669), `do` (5,631), `te` (5,558), `la` (5,062), `me` (5,021), `ho` (5,014), `train` (4,841), `er` (4,688), `re` (4,541), `mo` (4,437), `After` (4,312), `li` (4,121), `shirt` (3,997), `sign` (3,975), `photo` (3,968), `make` (3,718), `look` (3,713), `ta` (3,704), `want` (3,704)

#### Longest units heard at least 20 times
`chest fits inside the `, `are there motorcycles?`, `box fits inside the `, `are there skateboard`, `please think it over`, `apple to John.⏎John `, `er fits inside the `, `are there giraffes?`, `are there vegetable`, `are there surfboard`, `s fits inside the `, `are there windows?`, `are there computer`, `are there stripes?`, `are there carrots?`

111,654 units, 670,000 utterances heard.


## 2026-10-25: curriculum design, habituation test, profiling

- **Life `full_s0` finished** at 1M moments. Final per-kind results are in `out/monitor/report.md`; the best kinds were story (+0.19 over the majority answer), reason, reading, activity and open questions (each about +0.05).
- **Curriculum design:** [CURRICULUM.md](CURRICULUM.md).
  - Stages from babbling to 4th grade, each gated by mastery of its own items plus retention of earlier stages.
  - Spaced review: 70% current stage, 30% earlier.
  - Leniency that shrinks stage by stage.
  - A stop verdict when a stage cannot be passed.
  - The language proposals are re-ordered by the stage that needs them.
  - Curriculum daemon started (`organic-curriculum`, `data/world/curriculum.jsonl`): items per stage and subject, each with 2-4 acceptable answers, judged for correctness and fit to the stage.
  - Budget: text world 140, curriculum 100, photos 200, reserve 60.
- **Habituation test running** (`experiments/habituation_pipeline.sh`): two fresh 100k lives in the frozen world, the ear's fixed habituation rule on vs off. The question is whether the orchestrator's own input adaptation (each input neuron subtracts its slow average) does the job without the rule.
- **Profile** (2k clusters, 41 ms per moment):
  - cluster affinity 12%;
  - visual cortex 16%;
  - per-call numpy overhead on small arrays (mean, norm, reduce) about 30%;
  - recomputing the list of word outputs each moment 5%.
  At 23k clusters a moment took ~120 ms, dominated by streaming the cluster identities (23k x 1,760 floats, 160 MB) through affinity several times per moment, which is memory-bandwidth bound.

## 2026-10-25: habituation is not needed as a rule

A controlled pair (`out/hab_on_s0`, `out/hab_off_s0`): fresh lives in the frozen world, seed 0, 100k moments, identical except for the ear's fixed habituation rule.
- Same reward (0.63 / 0.63) and the same number of clusters (31,685 / 31,808).
- The same segmentation.
- Per kind within noise: off was +0.10 on yes/no, +0.07 on story and reading, -0.05 on color and talk.

The orchestrator's own input adaptation (each input neuron subtracts its slow average) does what the rule did. **The rule is now off by default** (`Ear.habituation = False`), so habituation comes from structure.

**Baseline curriculum life started** (`out/cur_s0`, via the `organic-life` service): today's brain through the stages, to get its verdict.

## 2026-10-26: first curriculum verdict, and the speaking brain

**Baseline curriculum life (`out/cur_s0`, today's whole-answer brain, cortex on):** *stops at W1 (first words): score 0.19 < 0.5 after 300,000 moments.*
- Stage B (echo) passed in the first 5k moments.
- W1 hovered at 0.17-0.21 for 300k moments. W1 is mostly naming what is in a photo (CIFAR-100, COCO's biggest object), which this eye-plus-cortex cannot do at 64x64.
- Stage B dropped from 0.83 to about 0.65 after promotion (at the edge of the 80% retention mark).

The verdict is honest and points at vision, not language, as the first wall.

**Speaking brain (`organic/speech.py`):**
- Stage B, first words only: 86% exact echo. Motor chunks formed ("thank you", "all gone"). Slips are rare and toddler-like.
- Copy route: nonword repetition rose from ~0 to 0.40 sound similarity. As a fixed route it caused echolalia; as a learned choice ("copy" vs "own words", credited by outcome) the brain learned to use its own words when the vocabulary was small (copy 5-14%).
- With a wide echo vocabulary (4,099 words from the text world), it learned to copy most of the time (71-88%). Nonwords are partly repeated ("tam!", "pug!", "sorn!"; sound 0.27). Exact echo is 0.31-0.35 and lenient 0.69: it repeats too much ("say sorn!", "juice, juice", with punctuation) because it cannot yet choose *which* heard unit to repeat or where to stop.
- Imitating every new word byte by byte grows clusters fast (19k clusters at 8k moments).

Next for speech:
- choosing which heard unit to repeat (an auditory attention shift, learned like gaze);
- stopping at the end of that unit.
Both are learned choices, not rules.

## 2026-10-26: kid-a, first release (L0, echo, like a 12-month-old)

- **New repo [kid-a](https://github.com/harrisonford/kid-a):** serves the brain, frozen, at every curriculum stage it has passed.
  - Svelte 5 chat app with images and a gaze overlay; history stays in the browser.
  - Standard-library Python API.
  - Runs at `http://100.98.161.113:8787` on Tailscale only (`kid-a` user service).
- **Training (`out/kid_s0`, `organic-life` service):** the speaking brain through the curriculum, cumulatively from stage B. Every promotion writes a candidate to `out/levels/`. The `kid-a-release` timer releases it as the next level.
- **L0 "echo":** passed stage B at 5,000 moments, with 0.94 lenient and 0.87 exact on held-out echo items. From the API:
  - "say blick" -> "blick", "say elephant" -> "elephant";
  - "banana!" -> "bananm", "say tomato" -> "toma";
  - "what is your name?" -> "remember".
- Curriculum change: first-words naming is limited to toddler nouns (dog, cat, car, cup, banana...); the rest of CIFAR and COCO naming moved to later stages.

## 2026-10-26 (evening): first words hit the vision wall; picture-book W1 and spatial cortex input

`kid_s0` (speaking brain, cortex pooled to one summary per layer) passed B (L0 released) and then stayed at W1 0.11-0.20 for 45k moments.

A nearest-neighbour probe (`experiments/w1_vision_probe.py`) gives roughly the ceiling of prototype clusters:

| naming task | 1x1, layers 2-3 (as used) | 2x2, layers 1-3 | 4x4, layers 1-3 |
|---|---|---|---|
| 34 toddler nouns (CIFAR) | 0.15 | 0.23 | 0.29 |
| ~11 picture-book nouns (CIFAR) | 0.35 | 0.43 | 0.54 |
| biggest thing in a COCO scene | 0.11-0.21 | 0.10 | 0.18 |

So the brain sat at the ceiling of its own visual features. Changes:

- **W1 is picture-book naming** of single objects (25 nouns; CIFAR). The other toddler nouns move to preschool. Naming the biggest thing in a busy scene moves to kindergarten: it is scene understanding, not a first-words skill.
- **The cortex reaches the orchestrator with coarse spatial layout:** 2x2 regions per layer, all three layers. Layer 1 arrives at tick 2, layer 2 at tick 3, layer 3 at tick 4. Options `cortex_grid` / `cortex_layers`; old brains keep 1x1 / layers 2-3. 4x4 would not fit in memory.
- **A fresh cumulative life `kid_s1`** (budget 30k clusters) replaces `kid_s0`. L0 stays the one released from `kid_s0`; the next levels come from `kid_s1`.

First sign in the new brain: asked "what is this?", it echoes "this?" (echolalia). The learned copy-or-own-words choice should outgrow it.

## 2026-10-27: a trace-rule IT layer does not help (negative result)

`experiments/trace_it_probe.py` tested an IT layer on top of the visual cortex:
- 600 competitive units, 20 active, reading the layer 2+3 code with a 2x2 layout;
- developed on 6,000 unlabeled CIFAR photos, each seen as 4 views (shift, flip, slight zoom);
- with the Foldiak/VisNet trace rule vs the same rule without the trace.

Nearest-neighbour probe on picture-book naming (16 CIFAR classes, majority 0.07):
- layer 2+3 code (as used now): **0.378**
- IT with trace: 0.170
- IT without trace: 0.195

The sparse competitive layer loses more than the trace gains. Not pursued for now.

Meanwhile `kid_s1` sits at W1 0.24-0.36 (pass 0.5) at 125k moments.
