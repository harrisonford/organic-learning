# The work loop (from 2026-11-19): state, queue, log

Kris, 2026-11-19: "fix and improve and even propose new ideas we can implement in a loop, also research for similar
studies of my architecture and ponder if it's relevant enough what we have currently to be able to publish in
arxiv or a journal somewhere; you can work very long on this."

**Read this file first at every iteration; update it last.** The earlier loop's log is docs/DAYLOOP.md (closed).
The loop's wakeups fire only while the session is alive: nothing a life needs may depend on the loop.

## Standing rules (Kris's, in force)
1. Organic first: no hand-wired behaviour. A mechanism that wires a signal to a behaviour to raise a score is
   artificial (the novelty drive was; retired). Then: is it data, training quality, or size?
2. What the brain does on the way (babbling, echo, routes, yes bias, the store's contents) is observed, never a
   target or a gate (docs/CRAFT.md, 2026-11-18).
3. Answers are judged for whether they are acceptable, lenient on form and strict on meaning; scores count above
   what a blind child gets (docs/JUDGING.md, 2026-11-19). **No pass mark** (Kris, 2026-11-21): new lives run
   `GATE=plateau`: the world moves on when a stage's curve stops rising. Stage W2 is called "pages".
4. Claims need 3 seeds and the 300-item tests; negatives are recorded as carefully as positives.
5. Every decision into the docs and memoize; commit and push; health check before any status.
6. Machine: keep 2 GB free; probes in capped units at Nice=15, at most one or two beside three lives; never kill by
   pattern. External requests carry no personal or project names. The OpenRouter key is never printed; Jev <= $1.50.

7. **No performance hacks; slow organic wins** (Kris, 2026-11-20: "NEVER FORGET we dont want to hack performance
   for the sake of it we want to build slow organic wins"). Nothing is proposed or built because it raises a number:
   not more filter kinds, not tuning to a benchmark, not a trick from the literature. The public benchmark is a
   ruler for checking what we claim; nothing in the brain is changed to move it. Size is the last question, asked
   only when a measured need points to it. (I had proposed 1,024 filter kinds for the benchmark: withdrawn.)
8. **Biology decides, not the score** (Kris, 2026-11-23 evening: "we want to improve ONLY organically, no
   performance hacking, in fact if there's something very biological that doesnt improve we implement anyways as
   it can make sense the longer we add biological mechanisms"). A mechanism that brains are well known to have is
   adopted on that ground, also when it moves no number today; it is still measured, taken out again and reported
   as what it did. The other half stands: nothing goes in because it scores. Sizes and architecture are revised
   from what is known of the brain and from recent work (Kris: a 2011 recipe is too old a ground), not from a
   search over scores. The learner stays on the processor (Kris: "its fine then use cpu").
9. **Iterate on the architecture in the loop; measure against the current paradigms; send the draft** (Kris,
   2026-11-24 10:35: "once you finish research and results you can iterate on architecture on a loop given you
   understand the philosophy on design go full organic learning over performance but we do measure performance
   and difference with current paradigms you can work on a loop and you send me the draft paper to my email").
   So: when the running results are read, the loop takes the architecture items one at a time (queue E 23–30),
   each after its organic check, adopted on its biological ground (rule 8). Every change is still measured, and
   measured beside what the field does now: on the CIFAR-10 stream beside the same-shape backprop and the
   streaming baselines, and in a life where a life is the test. The paper's draft goes to Kris by email
   (the address used for drafts 1–5) when it has changed materially; not on every round.
10. **All kinds of real pictures, not only CIFAR** (Kris, 2026-11-25 20:20: "should use all type of pictures
   honestly why only cifar those pictures are shit we should have also wiki pics etc"). Vision is measured, and
   the cortex developed, on real pictures at the eye's size from several sources (COCO, Wikimedia Commons, more
   when we have them). CIFAR-10 stays as the row that can be compared with the literature, not as the only or
   the headline evidence. First step: the photo stream (experiments/bench_photos.py; `SET=photos` in
   bench_stream.py). My reading, to be confirmed by Kris: this settles "which size is the paper's benchmark"
   for the 64-pixel eye fed real pictures.

## Keeping the loop alive (2026-11-24)
- **Queues share one memory gate (2026-11-26):** several queue scripts gating on "3.3 GB available" can all pass in the
  same minute when a job ends and start together (four were waiting at once on 2026-11-26). New queue scripts
  `source out/tmp/gate.sh` and start each job with `gated CMD`: a lock all queues share, the room check, the job,
  five minutes holding the lock while its memory grows. Older queues still running keep their own gate; chain a new
  queue behind them (`while systemctl --user is-active -q <unit>; do sleep 120; done`) instead of beside them.
The probes and lives are systemd user units with lingering on: they run without any terminal. The loop itself is
the Claude Code process, and it dies with the terminal it was started from (a dropped SSH connection stopped it
from 01:42 to 10:08 on 2026-11-24). So Claude Code runs inside `tmux` on the desktop (`tmux new -s organic`;
detach with Ctrl-b d; come back with `tmux attach -t organic`). After any restart of Claude Code (`--continue`
or `--resume`) the self-paced loop is not restored: the `/loop` line has to be typed again. A reboot of the
desktop ends the tmux session too.
herdr (v0.9.5, installed for the user on 2026-11-24 at Kris's request, from its GitHub release, checksum checked) does
the same as tmux with its server on the desktop: `herdr` starts or reattaches, Ctrl-b q detaches.

## Running
- **`organic-relaysrc`** (experiments/relay_by_source_probe.py; 2026-11-27 10:50; MemoryMax 2500M; about 25 minutes):
  how even the lives' own relay's code is on each source they name (COCO crops, CIFAR, Commons), the delta reader
  against ridge, and the relay settled again → out/bench/relay_by_source.log. Said before: VISION.md 10:49.
- **`organic-relaylearn-s1`, `-s2`** (experiments/core50_relay_learns_probe.py; 2026-11-27 09:22; MemoryMax 2000M): the
  relay circuit learning along CORe50's stream, seeds 1 and 2 → core50_relay_learns_s{1,2}.log (middle fixation due
  about 11:40). Seed 0 read (VISION.md 10:49; `organic-relaylearn` stopped at 10:43 after it).
- **`organic-grounded2`** (out/tmp/grounded_queue2.sh; 2026-11-27 08:05; MemoryMax 2400M; about seven hours): the
  grounding run again (the first, `organic-grounded1`, was killed for memory at 07:49 after seven hours, coding the
  photographs in one stack). The probe now keeps the captions' bags in out/bench/grounded_bags_*.npz and codes the
  photographs 500 at a time; its resume was checked (a run that reads the bags back gives the same lines). → grounded_s0.log.
- **`organic-delta-c50`** (out/tmp/delta_c50_queue.sh; 2026-11-27 07:14; shared gate): the delta reader done on three
  seeds (read, VISION.md 08:28); the word cells' no-name count on seed 0 still to come (…_noname_s0.log).
- **`organic-stream-eye2`** ENDED about 07:55; three seeds read (VISION.md 08:29). **`organic-cells-err`** and
  **`organic-ncm-c50`** ENDED, read (VISION.md 07:11). Diagnostics run and read 08:15–08:28 (not gated, capped):
  core50_reader_norm.log, core50_spectrum.log.
- **`organic-fovea-q`** ENDED about 05:50, read (VISION.md 06:03). (out/tmp/fovea_queue.sh; queued 2026-11-27 03:19; shared gate): the eye on the thing (VISION.md
  02:55): CORe50 through a fixation, seeds 1–2 (core50_fovea_s{1,2}.log; codes core50_c50{centre,salient}_s*), the
  stream of 10 kinds on them (stream_core50_c50{centre,salient}_s{0,1,2}.log), the slow stage on the middle
  fixation, seeds 1–2 (core50_slow_centre_s12.log). **`organic-fovea-mv0`**: the eye drawn by what moves, seed 0
  (core50_fovea_moving_s0.log; VIEWS=moving): if it matches the middle, queue it on seeds 1–2 and its streams.
- **Gaze lives:** all three ended (gazs_s0 at 130,000 about 08:20: first words 0.47, two words 0.48, pages 0.20); in
  ended.json. The controls run in `organic-tag2` (tag_queue2.sh: tagn then tagr for seeds 1, 2, 0, then tagb):
  tagr_s1 running since 08:10 (added to lives.json by hand at 08:30; the queue does not add them: add each as it starts). All three now in out/monitor/lives.json; the ended ones in ended.json. The controls tagn_s2, tagn_s0
  run in `organic-tag2` after this queue.
- **SpokenCOCO through the cochlea** (`experiments/world/spokencoco_cochlea.py`, python -I, streams the archive): the
  validation part done 14:30 (25,035 recordings, 10.8 M rows: about 30 hours of speech) → ~/datasets/spokencoco/
  cochlea_64.f16 + cochlea.jsonl (transcripts kept for measuring only). All 118,287 training images of SpokenCOCO are
  among our COCO thumbnails (its val images are not). PART=train when the download is whole.
  **`experiments/sound_units_probe.py`** (hearing cortex → prediction → units from sound; measured by the
  transcripts only at the end): first design failed its smoke test, redesigned in three levels (VISION.md 14:45);
  `organic-sunits1` ENDED, read (VISION.md 16:30: syllable scale right, nothing recurs). **`organic-sunits2`**
  (out/tmp/sunits_queue2.sh, queued 16:40): speaker adaptation, 8 band stretches, two- and three-syllable window kinds,
  the lift reading → sound_units_val2_s0.log (about 80 minutes).
- **`organic-gaze-lives`** (out/tmp/gaze_queue.sh; queued 13:30; one life at a time at 5.7 GB available, so after the
  ramp lives): queue E 34 (a), the serial gaze in grown lives: gazs_s1/s2/s0 continued from plat_s*_W2 for 20,000
  moments of preschool (`GAZE=serial`, otherwise the tag queue's line); control: tagn_s* (tagn_s1 done 11-25).
  Records out/gazs_s*.jsonl; notes out/tmp/gaze_queue.log. READ THE FIRST LIFE'S LOG when it starts (first
  continuation from a level with the serial gaze switched on).
- **`organic-core50-1`** (out/tmp/core50_queue.sh; queued 13:55, behind organic-ncm1; MemoryMax 1500M; each step at 3.3 GB available):
  queue E 31 (b), real video for the eye: smoke test (stops the queue if it fails) → per seed core50_trace_s{S}.log
  (second stage in order / shuffled / plain, first stage alone: same object from another session) →
  stream_core50_c50{first,order,shuffled,plain}_s{S}.log (10 kinds, word cells and ridge). About 3.5 hours a seed.
  Prediction: VISION.md 13:20. CORe50 frames done 13:29 (164,866, the dataset's count; contact sheet checked: kinds
  in the order assumed, objects turning in the hand, other floors and light in other sessions).
- **`organic-spokencoco-fetch`** (started 12:50; ~/datasets/spokencoco/fetch.sh; 3 MB/s, resumable, 64 GB, about six
  hours): spoken COCO captions for the sound ear (queue E 32; docs/WORLD.md). Ends with `tar` in fetch.done.
- **`organic-av2`** (out/tmp/av_queue2.sh; queued 16:20; shared gate): two senses. Seed 0's which-sound is read
  (VISION.md 15:45); its word test was killed at the cap at 15:19 (a 900 MB array joined twice), now block-wise and
  run from the cached codes (av_probe_s0_what.log, WHAT_ONLY=1); then seeds 1 and 2 in full (av_probe_s{1,2}.log).
- **`organic-photos-bp3`** (out/tmp/photos_backprop_seeds.sh; queued 10:50; waits for relayshrink1, then 3.3 GB a
  run): backprop on photographs, seeds 1–2, relay and no relay, at seed 0's settings → backprop_photos_full.log.
  Fills the paper's pending mark.
- **`organic-ncm1`** (out/tmp/ncm_queue.sh; queued 11:50; waits for census2, then 3.3 GB a run): nearest class mean
  through the relay on the photo stream, ph_r and ph_d, seeds 0–2 → stream_photos_{ph_r,ph_d}_ncm_s{0,1,2}.log.
  The reader's diagnosis (VISION.md 11:40).
- **`organic-census2`** (out/tmp/census_queue2.sh; waits for census1): plat_s0's census again with turnover.
- **`organic-census1`** (out/tmp/census_queue.sh; queued 11:25; waits for av1, then 3.3 GB a life): the census of
  kind_s0..2, plat_s1..2 (now with word-cell turnover from the cells' birth moments), then docs/GROWTH.md again.
  plat_s0's census predates the turnover columns: run it again after.
- **`organic-video-audio`** (permanent service since 2026-11-26 09:50; `experiments/world/video_audio.py`; log
  out/video_audio.log): the kept Commons videos' sound through `organic/cochlea.py`, aligned ten frames to one
  picture frame → `~/datasets/commons_video/audio_64.f16`, `audio.jsonl`. Looks for new videos every hour.
- **`organic-relayshrink1`** ENDED, read (VISION.md 11:40: the floor is not the reason). (out/tmp/relay_shrink_queue.sh; queued 09:10, waits for relaymore1; gated): the relay with
  its floor at 0.0001 (not 0.0005) for ph_d and ph_r of seed 0 → relay_shrink_{ph_d,ph_r}_s0.log,
  stream_photos_{ph_dz1,ph_rz1}_s0.log. What I said before: VISION.md 09:10. A diagnosis, not a setting.
- **`organic-commons-video`** (permanent service since 2026-11-26 08:15; `experiments/world/commons_video.py`; log
  out/commons_video.log): Commons videos as 64 × 64 frames in order into `~/datasets/commons_video/` (raw files kept). **Since 2026-11-27 09:22 it waits while the disk has less than 100 GB free** (see the decisions).
  Kris asked for it ("download in parallel wikimedia"). To do: count real footage against animation by category
  before use.
- **`organic-core50-fetch`** (started 08:20; ~/datasets/core50/fetch.sh; curl limited to 2 MB/s, resumable): the video
  set for queue E 31; ends with `zip` in ~/datasets/core50/fetch.done.
- **`organic-relaymore1`** ENDED, read (VISION.md 09:10, 09:55, 11:10). (out/tmp/relay_more_queue.sh; queued 07:55, waits for photos2; MemoryMax 1800M; gated;
  about 1.5 hours): the relay settled on 20,000 photographs for ph_d and ph_r of seed 0 → relay_more_{ph_d,ph_r}_s0.log,
  stream_photos_{ph_dm20k,ph_rm20k}_s0.log. The hypothesis and what I said before: VISION.md 08:00. (Smoke test on
  300 photographs: 386 directions through the relay of 6,000, 198 through one of 300: the count moves as expected.)
- **`organic-photos-bp2`** ENDED, read (VISION.md 11:10: nothing wider wins). (out/tmp/photos_backprop_wide.sh; queued 07:50, waits for photos-bp1): the photo
  reference's tuning widened past its edge; the whole stream again if a wider setting wins.
- **`organic-sailp1`** (out/tmp/sailp_queue.sh; started 00:42; MemoryMax 1500M; gated; about five hours): SAILnet in
  the cortex's place on the photo stream, seed 0 → out/bench/sailp_develop_s0.log, sailp_code_s0.log,
  stream_photos_{sailpu,sailp}_s0.log. Beside: ph_r / ph_d of seed 0 (VISION.md 00:40, with what I said before).
- **`organic-photos2`** (out/tmp/photos_queue2.sh; started 00:30; MemoryMax 1800M; gated; about 2.5 hours a seed):
  the photo benchmark's seeds 1 and 2 → photos_code_{ph_r,ph_d}_s{1,2}.log, stream_photos_{ph_r,ph_d}_s{1,2}.log.
- **`organic-commons`** (permanent service since 2026-11-25 22:26; `experiments/world/commons_depicts.py`; log
  out/commons_depicts.log): Wikimedia Commons photographs by what they depict, one every two seconds, into
  `~/datasets/commons_more/` (never into `~/datasets/commons`: world 4). Kris asked for it ("slow polite
  download constantly"). To do with it: a contact sheet and a count by word after a day; decide how its
  pictures enter the photo stream (labels are noisy: "depicts" is not "is a picture of"); giraffe's item.
- **`organic-photos-bp1`** (out/tmp/photos_backprop_queue.sh; started 20:50; MemoryMax 1400M; gated at 3.3 GB a run;
  waits for cortex_ph_d_s0.pkl from photos1): backprop of the same shape on the photo stream, seed 0
  (`SET=photos` in bench_backprop.py, default unchanged; the code path has not run yet: READ
  backprop_photos_tune_s0.log FIRST for a traceback). Tuning on the first 10,000 (rate 0.0003 / 0.001 / 0.003 ×
  batch 8 / 32, no relay and fixed relay: twelve runs) → out/bench/backprop_photos_tune_s0.log; then the whole
  stream at the best of each → backprop_photos_full.log.
- **`organic-photos1`** (out/tmp/photos_queue.sh; started 20:35; MemoryMax 1800M; gated at 3.3 GB a stage; about
  three hours): the photo benchmark, seed 0 → out/bench/photos_code_{ph_r,ph_d}_s0.log (near copies, the code's
  reading), stream_photos_{ph_r,ph_d}_s0.log (50,000 pictures seen once, 4,000 tests, 33 words; word cells,
  ridge, streaming LDA, with and without the relay). What I said before it ran: VISION.md 20:30. Then: seeds
  1–2; backprop of the same shape on the same stream (bench_backprop.py needs the same `SET` switch).
- **`organic-sailn2`** (out/tmp/sailn_queue2.sh; started 19:47; MemoryMax 1400M; gated at 3.3 GB a stage; about
  50 minutes a seed): SAILnet at native size on seeds 1 and 2 → sailn_develop_s{1,2}.log, sailn_code_s{1,2}.log,
  stream_{sailn,sailnu}10k_s{1,2}.log. Beside them: nat_r / nat_d of the same seed.
- **`organic-sailn1`** ENDED 19:38, read (VISION.md 19:50). (out/tmp/sailn_queue.sh; started 18:52; MemoryMax 1400M; about 1.5 hours; gated at 3.3 GB a
  stage): SAILnet in our cortex's place at CIFAR's own size, seed 0 → out/bench/sailn_develop_s0.log (about
  55 minutes; different filters, rates), sailn_code_s0.log, stream_{sailn,sailnu}10k_s0.log. Beside: nat_r
  0.626, nat_d 0.649. What I said before it ran is in VISION.md 18:50.
- **Lives `kind_s0..2`** (organic-kind-s0/1/2; started by `organic-reward-queue` when the plateau lives ended,
  2026-11-24): the reward lives (JUDGING.md, 2026-11-23); `ramp_s0..2` follow from the same queue.
- **`organic-sh7`** (out/tmp/softhebb_dissect3_queue.sh; started 06:05, about 3.5 hours; MemoryMax 1500M): the
  dissection's five arms on seeds 1 and 2 → out/bench/softhebb_dissect_s{1,2}.log, then
  stream_{shh,shp,shc,she,sho}10k_s{1,2}.log.
- **`organic-softbase2`** (out/tmp/soft_baselines2_queue.sh; MemoryMax 1800M; waits for softbase1 and native4, each run
  for 3.3 GB available; about six hours): backprop started from the developed soft filters, with and without
  relay, and frozen filters without relay at power 0.7; three seeds → out/bench/backprop_soft_full2.log.
- **`organic-native4`** (out/tmp/native_queue4.sh; MemoryMax 1500M; every step waits for 3.3 GB available): the
  native-size control on seeds 1 and 2 (native_code_s{1,2}.log, stream_{nat_r,nat_d}10k_s{1,2}.log), then the same
  script on blown-up pictures, three seeds (big_code_s*.log, stream_{big_r,big_d}10k_s*.log). It may wait for
  hours behind softbase1 and tagn_s1.
- **Lives `ramp_s0..2`** started 12:20 (REWARD=ramp; 120,000 moments, about 26 hours). **`organic-tagn-s1`** (the
  first arm of tag1) started 11:38; no checkpoint yet at 13:00.
- **`organic-sailc1`** (started 09:19; MemoryMax 900M; about 45 minutes): SAILnet in our cortex's place, seed 0 →
  out/bench/sailc_develop_s0.log, sailc_weights_s0.npz. **`organic-sailc2`** (out/tmp/sailc_queue.sh; MemoryMax
  1500M; waits for sailc1 and sail2): codes and streams → sailc_code_s0.log, stream_{sailc,sailcu}10k_s0.log.
  **`organic-scan1`**: one free vision model tried as the photo world's second adult → out/model_scan.log.
- **`organic-sail2`** (out/tmp/sail_queue.sh; MemoryMax
  1500M; waits for sail1 and bcm3): its codes and streams → sail_code_s0.log, stream_{sail,sailr}10k_s0.log.
- **`organic-bcm3`** (out/tmp/bcm_queue.sh; MemoryMax 1500M; waits for sh7 and bcm4): codes and streams of the
  BCM layer at 250,000 / 2,500,000 / 10,000,000 presentations and as born → bcm_code_s0.log,
  stream_{bcm,bcm_2500000,bcm_10000000,bcmr}10k_s0.log.
- **`organic-softbase1`** (out/tmp/soft_baselines_queue.sh; started 04:45; MemoryMax 1800M; about
  eight hours): the references for the soft layer. In order: the whole stream coded (soft_codes.log; variants
  `shfull`, `shrfull`); the way back through the relay checked (backprop_soft_check.log: READ IT FIRST, the two
  columns must agree); backprop of the layer's shape tuned on seed 0 (backprop_soft_tune_s0.log); per seed the
  whole stream on the developed code (stream_shfull_s{0,1,2}.log: word cells, ridge, streaming LDA, with and
  without relay) and backprop at its best settings (backprop_soft_full.log); then stream_shrfull_s{0,1,2}.log.
  **ENDED 17:26, all read (VISION.md 14:55, 16:55, 17:45).** State 14:55: everything done and in the paper but the last part (the layer as born on the whole stream,
  stream_shrfull_s{0,1,2}.log, about an hour a seed, ends about 17:45). When it is in: the as-born row into the
  paper's second table, build, email draft 9.
- Finished and read tonight: sh6 (own / early, seed 0), sh5 (the dissection, seed 0), mkcircuit1 (the circuit for a life), relay3 (the relay circuit, three seeds), sh4 (the soft rule, seeds 1–2), sh1/sh2 (the soft rule, seed 0), relay1 (first run, wrong step), relay2 (the
  direction-by-direction reading), devcodes1, rpe1, trace2.
- **`organic-tag1`** (out/tmp/tag_queue.sh; never edit while it runs): queue E 29e, the life-scale test of replay
  marked by reward. Lives continued from the plateau lives' levels as they left pages, 20,000 moments of
  preschool, one at a time and only with 5.7 GB available (so it waits behind the reward lives): `tagn_s*`
  (nothing), `tagr_s*` (`REPLAY=0.3 REPLAY_TAG=reward`), then `tagb_s*` (`REPLAY=0.3`); units
  organic-tagn-s1 … ; notes in out/tmp/tag_queue.log; records out/tag{n,r,b}_s*.jsonl. The first arm is also the
  smoke test of continuing from a level with today's code: look at its log when it starts.
- Ended as planned: lives `plat_s0..2` (120,000 moments; read in JUDGING.md, 2026-11-24). Finished and read: the
  lateral network on three seeds, backprop with the relay, the gaze probes gaze1–gaze6, norm1, trace1, diet1.
  Also finished and read: rest1 (normalisation seeds), diet2 (blurred start seeds), expand1 and expand2.
- Finished and read: the clean and cleanw lives, the five ablation arms, the benchmark probes, the readout control.
- Launch line for a life with no pass mark:
  `TAG=plat SEEDS="0 1 2" ARM=routes WORLD=4 MAX_GB=5 TOTAL=120000 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1 -E GATE=plateau" experiments/conv_pipeline.sh`

## Queue (in order; the paper's needs are in docs/PUBLISH.md)
**A. The controlled core on a public benchmark** (offline; decides whether the efficiency claim survives)
1. `bench_stream.py` on seed 0: word cells with and without relay and store against nearest class mean, streaming
   LDA and ridge, developed and random filters; accuracy against pictures seen; one-shot at a lag of 20.
2. Online backprop of the same shape on the same stream, tuned for one pass (learning rate, batch size), and with
   replay at matched updates (extend `backprop_reference.py`).
3. The episodic store on raw and on backprop features (does the fast/slow dissociation need our cortex?).
4. Five seeds (cortex seeds 0–2 at least); CIFAR-100 if CIFAR-10 holds.
**B. The life**
5. Read the clean lives: do three seeds pass first words and two words above the floor with nothing artificial?
6. The loss at a stage change: compare a two-words level with its +5,000 checkpoint answer by answer (which answers
   replace the old ones); then the stage mix (70% new, 30% review, halving with each stage back): what would a
   home be like where naming never stops? Training program before mechanism.
7. Text is at the parrot's level (above the floor: 0.00–0.12): look at what the word cells' context side learns
   from a text lesson, item by item, before proposing anything.
8. Preschool's test is mostly answerable blind (floors 0.5–0.75): rebuild the stage's sources before any brain is
   asked to pass it.
   **Surveyed 2026-11-23** (VISION.md, evening; `experiments/preschool_sources.py`): five faults, a proposal for
   world 5 (numbers as words, built sources drawn evenly, conversation lived and not tested). **Built the same
   evening: `WORLD=5`** (world 4 unchanged item for item: `experiments/world_hash.py`). Open: a life in it.
**D. Owed to the paper (the referee's list, 2026-11-23; controls on the references and the ruler, nothing in the brain)**
16. An end-to-end backprop reference with a whitening layer before its readout (the like-for-like comparison).
17. The lateral network on the whole stream, three seeds; the same rule among the summary's 4,096 numbers.
18. The benchmark tested on all 10,000 test pictures; streaming LDA without the relay on seeds 1 and 2.
19. How often a picture comes again in a life; a life with no going back to earlier stages; the shuffled order.
20. A picture-blanked learner as one more blind baseline; a person's labels on 200 judgements (Kris).
21. The new-class test with even counts, altered views of the shown picture, and beyond the store's capacity.
22. Done 2026-11-23: a drawing of the learner (`fig_architecture`, figure 1), a life's curves and examples (the
    showcase section), and an appendix with the settings, the compute and the judge's instruction.
**C. Ideas proposed (each needs the organic check written down before it is built)**
9. **Next to build (as a probe first):** lateral anti-Hebbian learning among the cortex's kinds: one local rule for
   what the batch relay does and for what development lacks (VISION.md, 2026-11-20 night: developed kinds are
   selective but 658 pairs are near copies). Measured first by copies and selectivity, not by accuracy.
10. A fast store keyed by the fixated thing, not the whole view: fast mapping of a thing on a page, and a place for
    a new-word bias to arise.
11. The store's surprise measure: it reads 0.84 for everything; without it no replay rule can choose.
12. The yes bias as a developmental observation beside children's (+0.24 to +0.34 at two; ours +0.10 at one level).
13. An ablation per organ in lives, and the same lessons in shuffled order (the paper needs both).
14. A person's labels on 200 judged answers (Kris), and scores corrected by the judge's measured error rates.
15. Pivot frames for two-word text (docs/RESEARCH_TWO_WORDS.md), only after 7 says what is missing.
**E. Kris's directions of 2026-11-23 evening (under rule 8; a survey first, then the organic check, then a probe)**
23. **A learned gaze.** Survey done (docs/RESEARCH_GAZE.md); plan and organic check in VISION.md (2026-11-23
    21:45). Done: the probe that the heard word leads the eye (three seeds); the serial gaze in the learner
    behind `gaze = "serial"` (life env `GAZE=serial`; default unchanged, checked by digest). Running: the small
    learner raised both ways (`organic-gaze1`). Owed: the same from a grown learner's checkpoint, both ways, when
    a life's worth of memory is free (about 1.6 GB); then lives with `GAZE=serial`. Later steps: a fast and a slow
    memory of a place's value; the fovea handing over whatever it is on (now only places that stand apart).
24. **The cortex's architecture and sizes from biology and recent work**, not from a 2011 recipe. Survey running
    (what is known of early visual cortex in numbers; local-rule vision models of 2020–2026). The same order.
25. **What recent papers (2022–2026) bring**, for the paper's citations and for mechanisms worth building. Survey
    running.
26. Both whitening steps by local rules (the paper's result rests on two batch stand-ins): the lateral network on
    the whole stream and among the 4,096 summary numbers (queue D 17), then the patches' whitening.
27. A faithful soft winner-take-all development of the filters (the published SoftHebb rule, unchanged), measured
    as the developed filters were: the paper's "Hebbian did no better than random" is so far a result about hard
    winner-take-all only (docs/RESEARCH_RECENT.md).
28. The relay by interneuron gains (Duong et al. 2023: online whitening with fixed synapses and gain-modulating
    interneurons, slow synaptic plasticity beneath), a published local mechanism, beside the lateral network.
29. **From the second survey (docs/RESEARCH_ARCH.md, 2026-11-24), in this order:** (a) a temporal term in the
    filters' rule: answer the next view of a thing as this one was answered (Halvagal & Zenke 2023), on the
    serial gaze's successive views; (b) a blurred start: the filters develop on waves, then on blurred pale
    pictures that sharpen (Lu et al. 2025); (c) critical periods in sequence when the cortex has stages;
    (d) surprise sets how much is learned (one broadcast surprise; errors divided by their own variance);
    (e) what comes back in sleep is tagged at the moment itself; (f) a sparse expansion before the word cells
    and as the store's key. Each after its organic check, as a probe, read first on what it does to the code.
30. **The eye's map without a hand-set weight** (Kris, 2026-11-24: "why fixed, research a better solution").
    Measured: the word's pull is 0.06 at the named place against a reflex term that varies by 0.19, so
    "0.5 × standing-out + pull" drowns the word (first look on the named thing 0.50–0.53; by pull alone
    0.81–0.85; any form free of the raw sizes 0.79–0.86). Survey running on how the brain combines them.

31. **Video: real dynamics for the eye** (Kris, 2026-11-26 08:05: "how good would this architecture be for video
    processing? because images are static but reality is more dynamic ... would videos give our visual layer the
    dynamism that images lack? and maybe find out some insights and blind spots we miss with simple images?").
    Organic check: passes on the strongest ground in this queue; temporal contiguity is how the ventral stream is
    thought to learn what goes together (Földiák 1991; Wallis & Rolls 1997; Li & DiCarlo 2008 changed IT cells by
    swapping objects across saccades; newborn chicks reared with only jerky object motion fail to build
    view-invariant recognition; Halvagal & Zenke 2023 the local rule). What we had: every temporal attempt so far
    (LPL in VISION.md Attempt 1, the slow-views layer, the trace-rule second stage of 29a) learned from views made
    out of a still (zooms, gaze points, a picture sliding one pixel): such sequences can only teach tolerance to
    shift and scale, never to rotation in depth, deformation, light or occlusion. Steps: (a) find real video we may
    use (licence and size checked before any download): a continual-learning set of handheld objects filmed as
    sequences, Wikimedia Commons videos through the polite fetcher, and, needing an agreement Kris would sign, a
    toddler's head-camera recordings; (b) the clean probe: the same frames developed in their order and shuffled
    (time is the only difference), read as the stream and as tolerance (the same object in another session);
    (c) only then the lives. Words: verbs and events have no still picture; video is where they could be grounded.
    **Kris, 2026-11-26 08:20: "yes lets include video then in the future ai can be used this model with streaming of
    data".** (a) done: **CORe50** (Lomonaco & Maltoni 2017; CC BY 4.0, checked in its repository and on its page):
    50 handheld objects in 10 categories, 11 sessions (8 indoor, 3 outdoor) with different backgrounds and light, a
    15-second video of each object in each session from the operator's eyes, 164,866 frames cropped to 128 × 128;
    5.9 GB, fetching slowly into ~/datasets/core50 (`organic-core50-fetch`, resumable, a few hours). Not taken:
    Toybox (110 GB, no licence found); SAYCam and BabyView (a toddler's head camera; Databrary's agreement and an
    institution's ethics approval: possible through the science partners, Kris's call). Wikimedia Commons video
    later (ffmpeg is installed). Next for (b): frames to 64 × 64 in order; the probe: develop on the sessions'
    frames in order and shuffled (the trace term and our rule), read by the stream on held-out sessions.

32. **Live streams, all senses at once** (Kris, 2026-11-26 09:30, after the step-by-step artifact: can the brain take
    a live stream and respond constantly; add real audio and video beside images and text; "interconnectivity of
    data streams should make a more reliable brain"; his hypothesis: human capability needs all the pieces together,
    interconnected, "but it must be nuanced"; movement later, since a human brain moves first, then babbles).
    My reading, given to him: the prerequisite is already true (everything learns online, nothing frozen, the gate
    decides when to speak); what is missing is (a) time: the world comes as turns (moments), not as a clock; (b) real
    sound: the ear hears text bytes, not audio; (c) streams teaching each other: only the heard word teaches the
    picture today. Order I proposed: a cochlea-like front end and audio-visual video (the Commons videos we keep
    have sound tracks) → every stream's cells predicting the others' by the same local rule → the world as a clock
    where every stream delivers its frame each tick → measure what each stream adds by taking it away. All pieces
    present early but each simple, maturing in an order (hearing before sight is sharp; movement before babbling),
    and every piece measurable by removal; movement for now is the eye's and the mouth's, a body later.
    **Kris, 2026-11-26 09:45: the test it is built toward.** An app of his own with camera, microphone, screen and a
    text chat, like a video call: he talks to it, and it keeps the conversation going, more dynamic than current
    models. Performance is pushed but organic first. Far off (a proof of concept, then an MVP); on this CPU the tick
    may be slowed down; real time comes with a GPU or the cloud.
    **Kris, 2026-11-26 12:50: "for audio what i want is not to transcribe to text then work on text similar to how
    words are just bytes audio can be fed in the ear develop it's own cortex with interconnectivity and just work".**
    So: no transcription anywhere inside the brain. Sound enters as sound: the cochlea (fixed, like the retina) → an
    auditory cortex developed by the eye's own local rule (as in experiments/av_probe.py) → its own relay → the same
    word cells, store and gate, and links to sight by local rules. The ear's units come from the sound itself, by the
    byte ear's principle (a unit starts where the next frame stops being predictable; recurring stretches become
    units; Saffran et al. 1996). Needs speech heard with what it is about: spoken captions of the COCO photographs
    (licence to check before any download); comparison: visually grounded speech learned by backprop (Harwath and
    colleagues). Later: a vocal tract the mouth learns to drive by babbling (until then it hears sound, speaks units);
    the judge hears the reply on the world's side (a caregiver listening, not the brain transcribing). The lives'
    text ear stays as it is; the sound ear comes behind a switch, probe first.
33. **How the brain grows, stage by stage** (Kris, same message: "progressive learning and evolution of the brain
    reporting size complexity and behaviour on different development stages"). `experiments/growth_report.py` →
    docs/GROWTH.md and out/growth/{tag}.json from the lives' own records. **First reading (nine lives), corrected the
    same morning by the census:** the *counts* are capped early: word cells and the orchestrator's clusters sit at
    about 1,800–2,000 from 5,000–10,000 moments, the store is full (2,000) from 5,000, the output neurons at 338 from
    15,000; a life then turns its cells over inside "a skull of fixed size" (organic/words.py: the least heard,
    longest unheard cells go, a tenth at a time). But the *numbers held* grow: `experiments/growth_census.py` on
    plat_s0's stage checkpoints gives 65.5 million at the end of babbling, 200.9 at the end of echo (the orchestrator
    appears whole, 121.8), 243.2 at the end of first words, 252.3 at the end of pages; the ear grows from 16.2 to
    52.1 million and the word cells from 32.6 to 60.4 (more cues on each cell), the visual cortex stays at 16.8. I had
    told Kris "the brain does not grow" from the counts alone: half true. A child's brain grows for years and
    prunes later; ours grows its synapses, not its cells, and switches a whole organ on at once. Owed: per-organ sizes and the turnover
    (cells forgotten and born per 5,000 moments) in the report, read from checkpoints or added to the records
    behind a switch; then the organic check of a capacity that grows with age.
    **Turnover, six lives (2026-11-26 12:15; census of plat_s0..2, kind_s0..2 from the cells' birth moments):**
    between the end of babbling and the end of echo (5,000 moments) about 1,000 of the ~1,900 word cells are born and
    as many forgotten, half the population; from echo to the end of first words (45,000–70,000 moments) about 1,050
    more, and in pages about 500–650 in 30,000–40,000 moments: turnover falls more than tenfold after echo and then
    holds at about 1.5 cells per 100 moments. The median cell's age goes from 58% of the cells' own time at echo to
    87–89% after first words: a stable core of old cells and a fringe that churns. The same on all six lives. The
    parallel to check: dendritic spines turn over fast in the young and are mostly stable in the adult cortex
    (Grutzendler, Kasthuri & Gan 2002; Holtmaat et al. 2005; to verify). In docs/GROWTH.md, a table per life.

34. **An asynchronous brain, many stimuli at once** (Kris, 2026-11-26 13:00: "also fixations how are they moving do they
    move at all? so the final brain architecture i expect it to be async in general with multi stimuli i hope we can
    steer that direction as human brains process things"). What I told him: in the running lives the eyes do not
    move in time (on a page the reflex finds up to four places that stand out and the word cells get them all at
    once, with the whole page coarsely; a single picture is seen whole); the serial gaze (one fixation a tick, the
    reflex's pull fading within a quarter second, then where the heard words pull, eye commands learned from
    reward; VISION.md 2026-11-23 to 11-24) is built, behind a switch, level with "all at once" on three seeds of
    a small learner (pages 0.57 against 0.55; first look on the named thing 0.68–0.70 against the reflex's 0.49),
    and in no life yet; its eye commands die out after the first look, it never moves within one object, and it
    has no pursuit (that needs video). Why the architecture suits asynchrony: local rules need no global forward
    and backward pass, so every organ can update when its own input arrives; a brain has no global clock and its
    areas integrate over a hierarchy of time scales (Murray et al. 2014; Hasson et al. 2008; to verify). Order:
    (a) the serial gaze in grown lives (continued from the plateau lives' levels, against a continuation without
    it) when the ramp lives end; (b) a clock-driven world: time in milliseconds, every stream at its own rate
    (sound 10 ms, video frames 100 ms, fixations 200–300 ms, speech as it is said), the gate free to speak at any
    time, the world not waiting for the reply; on this CPU a discrete-event queue ordered by time (asynchronous
    in meaning, slowed down); (c) organs on their own time scales, each updated only by its own events; (d) every
    stream measured by taking it away, and by how soon an answer comes when sight and sound agree.
    Design written (docs/ASYNC.md, 2026-11-26 15:15): what is synchronous today, one timeline in milliseconds,
    rates and time constants from biology (to verify), five switched steps; step 1 is hearing in time within a moment.
    **Kris, 15:15: neurons, not turns** ("every neuron ticks and send info ... stream in stream out"): the main road is
    now a small stream brain on a 10 ms clock (organs as populations with their own time constants; outputs as
    populations racing to threshold; only changes propagate), first shown to look, hear and vocalise at once on
    Commons video with sound, then the learned pieces moved in one by one (docs/ASYNC.md).

## Decisions waiting for Kris
- **(2026-11-27 10:10, partly answered) The disk.** Kris, 10:05: "limpia todo lo antiguo que no es usado en brains
  definitivamente los october brains". **Deleted 10:08–10:10** (lists kept in out/monitor/deleted_2026-11-27_*.txt;
  logs and score files kept; nothing had them open; kid-a serves from its own copies, answering 200 after): the 91
  October brains in out/ (109 GB), the 8 October level files (14.5 GB), and the brains from before the review of
  2026-11-10 that nothing refers to (csize, overheard, hub, fz, the smoke tests' levels and four smoke-test brains
  in out/tmp: 67 files, about 29 GB). **Kept:** kid_s3 (the lineage kid-a publishes: 8.5 GB and 22.4 GB of levels),
  every life since the review, the caches in out/tmp. Free: 133 GB → 277 GB. Five old analysis scripts default to
  October brains and no longer run as they are (logic_ablation, ear_breaks, organ_view, raw_specialization,
  timed_analysis; their results are in the docs). **Still yours to decide:** the SpokenCOCO archive (68 GB), the
  Commons videos' downloaded files (57 GB; the service waits under 100 GB free and grows 6 GB an hour), the COCO
  zips (20 GB), three chat models nothing uses (8 GB), and the ended lives' brains since the review (about 160 GB).
  **10:19, on Kris's word ("borra también los kid_s0–s2 y los duplicados de Downloads"):** ~/Downloads emptied (4.3 GB):
  its film (twice) and album were abandoned partial downloads, mostly zeros and unplayable (checked block by block;
  the whole film is in /srv/video, the album in /srv/music). **A correction:** I had listed "kid_s0–s2 levels dated
  November (about 14 GB)"; there were none: those levels were the October files already deleted (I counted them
  twice). Only kid_s0–s2's logs and score files remain, and they stay. Free: 286 GB.
- **(2026-11-25 18:50, sharper) Our development helps at the pictures' own size, on three seeds** (word cells +
  relay 0.649 ± 0.008 against 0.626 ± 0.002 as born; ridge + relay 0.666 against 0.632). The paper's "did
  nothing" was on enlarged pictures. I will put both sizes side by side in that section when the enlarged
  rows on the same pictures are in; which one is the headline is the question below.
- **(2026-11-25) Which size is the paper's benchmark?** The benchmark shows CIFAR enlarged from 32 to 64 pixels
  because the learner's eye is 64 × 64; a 5 × 5 window then holds less than half of what a window of a real
  picture holds, and on seed 0 our own development helps at the pictures' own size where it did nothing
  enlarged. If three seeds agree, either the paper keeps "the learner as it lives" and says what the
  enlargement costs, or its benchmark moves to 32 pixels (a cortex the lives do not use). Until you say: both
  are measured and the paper keeps the enlarged benchmark with the caution.
- **(2026-11-25) Should lives that are compared between generations run in a frozen world?** A world is read as
  it is when a life starts, and the text and curriculum files grow while the daemons run, so the plateau, reward
  and ramp lives each started in a slightly larger world. A frozen copy can be given with `WORLD_DIR`. Until you
  say, the lives stay as they are and the comparisons carry the caution.
Kris, 2026-11-23 17:10: "on your questions i follow your decision you can decide for me". Decided by the loop:
- **The store keyed by what the eye is on: adopted for new lives** (`-E KEY=looked` in the life script; running
  lives stay as they are). Reason: three seeds, takes the store's harm on pages away and lets it keep a name heard
  once on a page; a limit stated (a memory of what was fixated, as it was fixated).
- **LaTeX installed** (pacman: texlive-basic, -latex, -latexrecommended, -fontsrecommended; pdflatex, as arXiv
  compiles). Next: the draft into LaTeX.
- **Licence for the preprint: CC BY 4.0** (told to Kris; Kris is at the licence step of the arXiv form).
Still Kris's own:
- The arXiv form: after the licence and policy steps, choose cs.NE as primary; that page says whether an
  endorsement is needed. Nothing is uploaded until the paper is ready and every reference has been opened by hand
  (docs/paper/REFERENCES.md).
- The author line (name, affiliation or none, public contact address) and the AI-use statement in Kris's words.
- Releasing a level to kid-a; hardware (memory first); labelling about 200 answers for the judge's audit set.
- **Queued behind the plateau lives: `organic-reward-queue`** (lives `kind_s0..2`, then `ramp_s0..2`; JUDGING.md
  2026-11-23). Lives in world 5 with the keyed store come after it.
- **Launch line for new lives** (when the plateau lives end):
  `TAG=... SEEDS="0 1 2" ARM=routes WORLD=5 MAX_GB=5 TOTAL=120000 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1 -E GATE=plateau -E KEY=looked -E RPE=normalised" experiments/conv_pipeline.sh`
  (`RPE=normalised` since 2026-11-25: the reward's error against its usual size, three seeds with control, VISION.md)

## Log
- 2026-11-19 22:30 Loop opened. Done this evening: judge audit and the strict judge; recast in teaching; blind
  floors and `above` in every record; GATE=above; echo a period; novelty drive retired; three literature searches
  (research_notes/Relatives and publishability 2/); docs/PUBLISH.md; three clean lives started; benchmark codes
  running; `bench_stream.py` written, not yet run.
- 2026-11-19 23:00 Clean lives at 5,000: babbling passed by seeds 0 and 2 (0.20, 0.19), seed 1 at 0.17. Benchmark
  codes for the developed cortex done (seed 0); the never-developed one is being coded. **Running:
  `organic-bench-dev0`** (`bench_stream.py`, developed, seed 0, 50,000 pictures → out/bench/stream_developed_s0.log;
  about 4 hours beside the lives; smoke at 1,000 pictures: word cells + relay 0.49, nearest class mean + relay 0.49,
  streaming LDA 0.50, ridge 0.49–0.51). **Queued behind it: `organic-bench-bptune`** (out/bench/tune_backprop.sh:
  `bench_backprop.py`, random filters, rate x batch on the first 10,000 pictures → out/bench/backprop_tune_s0.log).
  Next: read both; run the best backprop setting on the full stream, with replay 1 and 4, and with the developed
  filters as start and frozen; `bench_stream.py` with VARIANT=random when its codes exist. When memory allows
  (3 GB): queue item 6 with `judge_audit.py` on out/eyes2w_s2.pkl and out/eyes2w_s0.pkl (the brains after 100,000
  moments of preschool) against their two-words levels, answer by answer.
- 2026-11-19 23:30 **Queue item 6 answered in part** (RESEARCH_MEMORY.md, last section): the answers a brain loses
  in preschool went (1) to preschool's own larger set of names for the same pictures (couch for bed, orange for
  apple) and (2) to the question said back, which the old judge rewarded. Not simple erasure. Next on it: the clean
  lives' `says_last` in preschool; a near-miss reading of the early tests.
  **Benchmark, first numbers (seed 0, developed cortex, CIFAR-10, one pass):** word cells + relay 0.395 / 0.491 /
  0.543 / 0.599 / 0.623 / 0.662 / **0.686** after 500 / 1,000 / 2,000 / 5,000 / 10,000 / 20,000 / 50,000 pictures;
  without the relay 0.581 at the end. The other arms are still running. The "seen once, 20 later" reading (0.84)
  is not the fast-mapping test (the classes are known by then): to be redone with a class held back from the stream.
  Clean lives: all three out of babbling (5,000, 10,000, 5,000). Random-filter codes done.
  Next: read `stream_developed_s0.log` and `backprop_tune_s0.log`; then `VARIANT=random`; then the full backprop runs.
- 2026-11-20 00:05 Clean lives: the echo period lived (0.94, 0.91, 0.84 at its one checkpoint), all three in first
  words from 10,000–15,000; naming above the floor already 0.03–0.07 from comments alone.
  **Benchmark, seed 0, developed cortex, after 50,000 pictures (5,000 in brackets):** streaming LDA + relay 0.741
  (0.619), streaming LDA 0.722 (0.623), ridge 0.709 (0.607), **word cells + relay 0.686 (0.599)**, word cells 0.581
  (0.486), nearest class mean + relay 0.567 (0.549), nearest class mean 0.312. So the word cells are 2–5 points
  under the best gradient-free linear readouts on the same code (which keep and invert a 4,096 x 4,096 matrix, and
  whose shrinkage was picked on the test pictures); the relay is worth 10 points to them. ridge+relay still running.
  Written: `experiments/bench_newclass.py` (a class held back, then shown 1 / 5 / 20 times: the benchmark's version
  of fast mapping; replaces the "seen once, 20 later" reading). **Queued, each waiting for the one before:**
  `organic-bench-bptune` (after dev0) → out/bench/backprop_tune_s0.log; `organic-bench-rand0` (after dev0) →
  out/bench/stream_random_s0.log; `organic-bench-newclass` (after rand0) → out/bench/newclass_s0.log.
  Next: read them; a benchmark section in VISION.md and the efficiency claim in PUBLISH.md rewritten on these
  numbers; full backprop runs at the tuned setting (replay 0, 1, 4; developed start; frozen).
- 2026-11-20 01:00 **Clean lives: seeds 0 and 2 PROMOTED out of first words at 15,000 moments** (5,000 in the stage;
  above the floor 0.27 and 0.29; beyond the parrot 0.44 and 0.46, so the old mark too). Seed 1 entered first words
  at 15,000. On entering two words both say "no" to every "is there" question (yes bias −0.53).
  **The measure refined** (JUDGING.md, last section): `above` is read on the questions a parrot cannot answer,
  beside the best of always-yes, always-no and the commonest answer. Correction: the brains do know some text
  (0.11–0.33 of such questions). Existing levels: first words 0.29 / 0.46 / 0.43, two words 0.23 / 0.35 / 0.32.
  Smoke life clean. **`organic-relaunch-clean`** (out/tmp/relaunch_clean.sh, one-shot) restarts each clean life one
  minute after its next checkpoint so that it runs the refined measure; log out/tmp/relaunch_clean.log.
  **Benchmark table, seed 0** → VISION.md (last section): word cells + relay 0.686; streaming LDA + relay 0.741;
  random filters as good as developed ones (0.691); backprop tuning 3 of 9 settings done (0.40–0.48 at 10,000).
  Still running: `organic-bench-bptune`, `organic-bench-rand0`; `organic-bench-newclass` waits for rand0.
  Next: finish reading those; full backprop runs at the best setting; PUBLISH.md's claims rewritten on the table
  (what is ours is no longer "beats other gradient-free readouts", and "development" is not a selling point).
- 2026-11-20 07:15 **The loop was silent from 01:00 to 06:55** (third time: the wakeups stop when the session drops;
  it does not run inside tmux). What ran by itself meanwhile:
  - **All three clean lives passed first words (15,000 / 20,000 / 15,000) and two words (25,000 / 35,000 / 25,000)**
    at the provisional mark: two words above the floor 0.26, 0.28, 0.27 (text 0.19–0.27, a thing on a page 0.26–0.34,
    "is there" 0.24–0.27), 10,000–15,000 moments in the stage. Levels: out/levels/clean_s*_W2.pkl. Then preschool,
    where nothing rose (above 0.00–0.03) and first words and two words slid (0.42 → 0.26; 0.27 → 0.19) as before,
    with no parrot's reward this time (`says_last` 0.02–0.10 throughout). **The mark of 0.25 let them leave two words
    when they had only begun it**: the earlier brains stood at 0.23–0.35 after 30,000–45,000 moments.
  - Benchmark seed 0 finished (VISION.md): backprop grid 0.40–0.53 at 10,000 (best at the grid's edge); random
    filters as good as developed; the new-class test (word cells 0.00, with the store 1.00; nobody reaches the kind).
  **Done now:** the three lives stopped in preschool at 45,000–50,000 (pickles kept: out/clean_s*.pkl; the stage's
  test is mostly answerable blind and was to be rebuilt anyway). **Started: `cleanw_s0..2`** (units
  organic-cleanw-s0/1/2) from the clean lives' first-words levels, held in two words (PASS_ABOVE=0.99) for the
  stage's 100,000 moments: how far does two words go, and where should its mark be? **Started:
  `organic-bench-queue2`** (out/bench/queue2.sh, about eleven hours, needs nobody; progress in out/bench/queue2.log):
  wider backprop grid → the best setting on the whole stream (replay 0, 1, 4; developed start; frozen) → seeds 1
  and 2 of codes, streams, backprop and the new-class test (→ backprop_tune_s0.log, backprop_full.log,
  stream_seeds.log, newclass_seeds.log).
  Next: read queue2 as it goes and build the three-seed table; the cleanw curves at each checkpoint; then queue
  items 8 (rebuild preschool's sources) and 9 (the relay learned online).
- 2026-11-20 07:20 Kris: an arXiv preprint alone may be the goal ("it could be just an arxiv publication right?");
  and a correction: no performance hacks (standing rule 7; the 1,024-kinds run is withdrawn, never queued). The
  wider backprop grid's first point (rate 0.003, 64 a step): 0.484, under the best so far (0.527): the grid is
  flattening near 0.5 at 10,000 pictures. The organic question the benchmark does raise: **why does the cortex's
  Hebbian development buy nothing over random filters?** (queue, before any thought of size).
  Plan for the preprint: when queue2's three-seed tables and the two-words curves are in, a first draft for Kris.
- 2026-11-20 23:40 **The loop was silent from 07:20 to 23:15** (fourth time; the queue and the lives ran alone, as
  planned). Collected:
  - **Benchmark, three seeds** (VISION.md): word cells + relay 0.687 ± 0.005 (random filters 0.693 ± 0.002); one-pass
    backprop of the same shape 0.616 ± 0.023, with one replayed picture per new one 0.661, with four 0.688; streaming
    LDA + relay 0.739; word cells without relay 0.591. The lead over backprop is the relay's. New class: as seed 0.
  - **Two-words curves** (VISION.md): flatten at 0.41 / 0.31–0.36 / 0.36–0.41 after 20,000–30,000 moments in the
    stage; first words keeps rising meanwhile (0.28–0.40 → 0.40–0.45).
  - **What development did** (`cortex_dev_probe.py`, VISION.md): kinds moved far, became selective, and 658
    pairs became near copies; nothing keeps kinds apart. Proposal with its organic check: lateral anti-Hebbian
    learning among the kinds (also the relay's local form). Not built.
  **Done:** docs/paper/DRAFT.md (first draft of the preprint, pending controls marked); `bench_readout.py` (control:
  is the lead the code's or the word cells'?) running; `LAST_STAGE` in the life script; two cortex twins built
  (never developed with its own relay; no relay); the ablation queue written and smoke-tested, started.
  Next: read `readout_*.log` and put the row in VISION.md and the draft; the anti-Hebbian probe on the benchmark's
  codes (copies and selectivity first); then queue item 8 (rebuild preschool's sources); read each ablation arm as
  it ends (out/tmp/ablation_queue.log).
- 2026-11-20 23:55 **Learned lateral inhibition, first form: negative** (`lateral_probe.py`, VISION.md): my own
  three rules made more copies than winner-take-all (1,377–2,453 pairs against 658) because the inhibition never
  settles (it grows without end). A fault of the formulation. Next: read the published rules (Földiák 1990;
  Zylberberg, Murphy & DeWeese 2011; Pehlevan & Chklovskii's similarity matching) in the papers and build the probe
  on one of them unchanged; table of copies and selectivity first. **Started: `organic-ablation-queue`** (waits for
  the cleanw lives, which end at about 02:00; smoke lives on both new cortices ran clean).
- 2026-11-21 00:30 **Readout control, seeds 0 and 1** (VISION.md, PUBLISH.md, the draft): a gradient softmax readout
  on the same relayed code does as well as the word cells (0.696 / 0.702 against 0.686 / 0.693). The lead over
  end-to-end backprop is the code's; the draft's abstract and first claim are reworded. Seed 2 and the random
  filters still running (`organic-bench-readout`).
  **Lateral inhibition on the published rules** (`lateral_probe.py` rewritten on Pehlevan & Chklovskii's
  Hebbian/anti-Hebbian network, equations read in the paper, arXiv:1503.00680): in a 600-picture smoke the
  inhibition stays bounded (mean 0.08, max 1.4) and the settled activities are decorrelated (top 1% of pairs 0.41),
  where my own rule ran away. **Running: `organic-lateral`** (16,000 pictures, seed 0 → out/bench/lateral_s0.log).
  Next: read its table (copies? selectivity?); only if the kinds are both selective and apart, CODES=1 and
  `bench_stream.py` with VARIANT=lateral, with and without the batch relay; the readout control's last rows into
  VISION.md and the draft; the cleanw verdicts (about 02:00) and the first ablation arm's start.
- 2026-11-21 09:15 **The loop was silent from 00:30 to 09:05** (fifth time; everything queued ran alone). Collected
  (VISION.md, 2026-11-21 morning): the readout control on three seeds and on random filters (the lead is the code's:
  gradient readout 0.694, word cells 0.687); the published Hebbian/anti-Hebbian network is stable and keeps the
  kinds' *activities* apart (0.40) but not their filters (1,204 pairs alike): the code to measure is the settled
  activity, not made yet; the cleanw lives ended as planned (two words 0.40 / 0.36 / 0.38 at the end, first words
  0.43 / 0.44 / 0.47); **ablation arm 1: with a never-developed cortex first words is the same, a thing on a page
  is not named (0.00–0.10 against 0.25–0.37, three seeds)**: development matters where the brain must look at one
  thing among several. Arm 1 ends at 50,000 (about 10:00); arm 2 (no relay) starts by itself.
  Next: `look2_probe.py` on the developed and the never-developed cortex (where does the gaze go; what does the
  fixated thing look like); the draft's "development buys nothing" reworded with this; the settled-activity code of
  the lateral network on a subset; then queue item 8.
- 2026-11-21 09:35 Kris asked what the "two words pass mark" means ("we're forcing the model to say a certain amount
  of words?"). Answered: no; it is a stage's name and the score at which the world moves on. **My recommendation,
  waiting for Kris's yes: no mark at all (the world moves on when the stage's curve stops rising), and the stage
  renamed "pages".** Not built.
  **Running: `organic-look-rand`** (`look2_probe.py` on the never-developed cortex, wide pages, seeds 0 and 1 →
  out/tmp/look2_random.log; then quarters+match on the developed one → out/tmp/look2_developed_quarters.log): the
  gaze itself is colour centre-surround and does not use the kinds, so the probe asks whether a fixated thing is
  named from never-developed kinds (developed: 0.47–0.53 on a page, 0.54–0.59 alone, 2026-11-17).
  **Queued behind it: `organic-lateral-queue`** (out/bench/lateral_queue.sh): the Hebbian/anti-Hebbian network
  developed again and kept (out/bench/lateral_s0.npz), the code made of its settled activities for a 10,000-picture
  stream, then `bench_stream.py` VARIANT=settled with and without a relay → out/bench/stream_settled_s0.log. To
  compare at 10,000 pictures: developed word cells 0.500 without relay, 0.623 with; streaming LDA 0.664.
  (`settle` was 60 times slower than needed under threaded BLAS: row layout and one thread.)
  Next: read both; the first ablation arm's last checkpoint and the no-relay arm's start.
- 2026-11-21 10:00 **Kris agreed: no pass mark, and the stage renamed.** Built: `GATE=plateau` (JUDGING.md, last
  section), W2 named "pages"; replayed on the held lives it leaves the stage after 40,000–45,000 moments at
  0.33–0.38; smoke life `organic-smoke-plat` started. Kris has an arXiv account. **Ablation arm 1 ended** (50,000
  moments): never-developed cortex, first words 0.41 / 0.41 / 0.41, pages 0.20 / 0.19 / 0.16 (control at 50,000:
  0.38–0.44 / 0.36–0.37). Arm 2 (no relay) started 09:33. Still running: `organic-look-rand`,
  `organic-lateral-queue`.
  Next: read the smoke, the gaze probe and the lateral queue; when the ablation arms are done, three lives from birth
  with `GATE=plateau` (the paper's staged-life numbers should come from the rule we keep).
- 2026-11-23 12:15 **Silent from 2026-11-21 10:00 to 2026-11-23 11:50** (Kris's own computer went off; the server did
  not: every queue ran to its end). Collected (VISION.md, 2026-11-23): **the ablation table, five arms x three
  seeds** (relay: the only part that costs first words; looking and the cortex's development: naming a thing on a
  page falls to 0.02 and 0.04; match neurons: "is there" on two seeds; the store: nothing these tests ask for, and
  page naming higher without it); the gaze probe does not reproduce the never-developed cortex's failure (0.37–0.45
  in the probe, 0.04 in the life): open; **the lateral network's settled activities give the word cells 9.5 points
  without any relay** (seed 0, 10,000 pictures). The draft has the ablation table.
  **Started: lives `plat_s0..2`** (organic-plat-s0/1/2) from birth with `GATE=plateau`, TOTAL=120000: the staged-life
  numbers under the rule we keep. Launch line:
  `TAG=plat SEEDS="0 1 2" ARM=routes WORLD=4 MAX_GB=5 TOTAL=120000 CORTEX=data/visual_cortex2.pkl CORTEX_LAYERS=0 EXTRA="-E CHAT=0.2 -E FULL=1 -E LOOKING=1 -E MATCH=1 -E EPISODES=1 -E INTAKE=1 -E GATE=plateau" experiments/conv_pipeline.sh`
  Kris is starting the arXiv submission form (endorsement check); the paper itself is not ready to upload.
  Next: (1) view-by-view diagnostic on out/ablrand_s0.pkl against out/cleanw_s0.pkl (why no page naming without
  development); (2) the store on pages (why higher without it); (3) related work, references opened one by one;
  (4) the lateral network on three seeds and the whole stream; (5) queue item 8.
- 2026-11-23 12:50 **View by view on saved brains** (`experiments/page_views_probe.py`, 100 test pages each;
  out/tmp/page_views.log, page_views2.log still running on eight brains): every brain names a fixated thing alone
  about as well (control 0.54, never-developed cortex 0.46, no store 0.51 of pages have a fixation that alone gives
  the right name), but as it lives the control names 0.36, the never-developed 0.15, the brain without a store 0.52.
  What it says instead is mostly the coarse whole page's answer. With the store silenced for the asking the control
  rises (seed 0: 0.36 → 0.48; seed 1: 0.45 → 0.49). In the code the store is written and read under the whole
  (coarse) view only (`organic/words.py`, `x[0]`): pages look alike coarsely, so it gives back another page's
  name. **Reading so far: it is the store's key, not the cortex's development, that fails on pages**; development
  only makes coarse pages less alike. Waiting for the never-developed brains' store-silent numbers before writing
  it into VISION.md. Then queue item 10 (the store keyed by what is fixated) gets its organic check and a probe.
  Lives `plat_s0..2` out of babbling at 5,000.
- 2026-11-23 13:30 **The store's key is what fails on pages: confirmed on three seeds** (VISION.md, 2026-11-23
  afternoon): with the store silent the control names 0.48–0.50 of test pages (0.36–0.45 with it) and the
  never-developed cortex 0.33–0.44 (0.07–0.15). The draft's reading of the ablation table is corrected (development
  is worth 0.05–0.10 on a fixated thing; the store as keyed costs page naming).
  **Built (off by default): `Episodes.key = "looked"`** (`organic/episodes.py`, `organic/words.py`): a moment is kept
  under the windows the eye fixated (the whole view when it fixated nothing) and compared with what the eye is on
  now. Organic check in VISION.md. Unit-tested. **Running: `organic-pageviews`** → out/tmp/page_views3.log: the three
  control brains and one never-developed, pages and single pictures, store as it is / silent / keyed by what the eye
  is on (their stores hold each moment's fixations, so nothing has to be lived again).
  Next: read it. If the keyed store does no harm on pages and none on single pictures: a probe of its own job on a
  page (a name heard once for a thing on a page, asked on another page with the same thing), then lives. Then
  related work for the draft.
- 2026-11-23 14:15 **The store keyed by what the eye is on does no harm** (VISION.md): on the saved brains pages are
  named as with a silent store (0.49 / 0.49 / 0.50; never-developed cortex 0.43), single pictures unchanged.
  Written: `experiments/fastmap_page_probe.py` (a name heard once for a thing on a page: asked on the very page, the
  picture on another page, the picture alone, another of the kind; word cells alone, store keyed by the whole view,
  store keyed by what the eye is on). Smoke (300 moments, keyed store): the very page 0.88, the picture on another
  page 0.88, the picture alone 0.16, another of the kind 0.00. **Running: `organic-fastmap-page`** (three seeds, 6,000
  moments → out/tmp/fastmap_page.log, about 1.5 hours). Lives `plat_s0..2` in first words at 15,000 (above the floor
  0.33 / 0.32 / 0.41); no stage left on a plateau yet (needs six tests in the stage).
  Next: read the probe. If the keyed store keeps a page's name and costs nothing: Kris's word, then lives with it
  (KEY=looked in the life script). Meanwhile: related work for the draft.
- 2026-11-23 15:10 **A name heard once on a page, three seeds** (VISION.md): keyed by what the eye is on, the store
  names the same picture on another page 0.85–0.92 (whole-view key 0.11–0.15; word cells alone 0.00); the picture
  alone and another of the kind stay at 0.00; old names untouched. Written: docs/paper/REFERENCES.md (28 references
  with links and how each is known, for Kris to open one by one). Kris asked what to do on arXiv: told (start the
  form in cs.NE to see whether an endorsement is asked; author line, licence, AI-use statement; nothing uploaded).
  Lives `plat_s0..2` at 20,000 in first words (0.35 / 0.33 / 0.41).
  **Waiting for Kris:** the keyed store in new lives (yes/no); installing LaTeX on the server (to compile the paper).
  Next: the plateau lives' first stage changes (this evening); figures for the paper from the logs; the lateral
  network on three seeds; queue item 8 (preschool's sources).
- 2026-11-23 16:00 Nothing new finished (lives at 25,000 in first words: 0.36 / 0.38 / 0.43, still rising; lateral
  network seed 1 being coded). Made: `experiments/paper_figures.py` → docs/paper/fig_benchmark.pdf/.png (accuracy
  against pictures seen, three seeds with their range; palette checked with the dataviz skill's validator, a marker
  per series). matplotlib added to the project's environment for it.
  Next: figure 2 (the stage curves) when the plateau lives have them; figure 3 (the ablation table as bars); read
  the lateral queue; the lives' first stage change (earliest at 45,000, about 19:00).
- 2026-11-23 16:45 **Lateral network, seed 1: reproduced** (VISION.md, evening): without any relay the word cells
  gain 9 points on its settled code (0.532 → 0.619; seed 0: 0.500 → 0.595); with the batch relay on top +1.9 (seed
  0: +4.7); streaming LDA unmoved. Seed 2 is being coded. **Paper figure 3** (`docs/paper/fig_ablation.pdf`): the
  ablation arms as bars with the seeds as dots, read back from the lives' logs (same numbers as the table); the
  draft points to figures 1 and 3. **Queue item 8 surveyed** (`experiments/preschool_sources.py`): preschool is 15
  sources taking turns, 20 test items each; numbers answered as digits in one source and words in two; three built
  sources lopsided (blind child 0.56–0.60); three conversation sources that no answer key can judge. Proposal for
  world 5 in VISION.md. Lives `plat_s0..2` at 30,000 in first words (0.41 / 0.39 / 0.43, still rising; no plateau
  yet). "Running" above rewritten (it still listed finished work).
  Next: build world 5's preschool (branch on `world >= 5`, a test that world 4 is unchanged item for item); figure 2
  when a life has left a stage; seed 2 of the lateral network and the three-seed table.
- 2026-11-23 16:50 **World 5 built: preschool rebuilt** (`organic/curriculum.py`, `world >= 5` only; VISION.md):
  numbers as words in every source, the three annotation-made sources even over their answers (blind child 0.25 /
  0.50 / 0.33 instead of 0.57 / 0.60 / 0.56), conversation lived but kept out of the test (`Curriculum.talk`).
  World 4's fingerprint is the same before and after (`experiments/world_hash.py 4 5`); the three lives are not
  touched. No life in world 5 yet: the machine is full with the three plateau lives.
  Next: figure 2 when a life has left a stage (earliest about 18:30); seed 2 of the lateral network (about 18:00); then, when the
  plateau lives end, lives in world 5 (and the shuffled-order control).
- 2026-11-23 17:20 **Kris leaves the open questions to the loop** ("you can decide for me"). Decided: the keyed
  store is adopted for new lives (`KEY=looked`, added to `experiments/curriculum_life.py`; running lives untouched);
  LaTeX installed with pacman (pdflatex); licence CC BY 4.0 recommended to Kris, who is at that step of the arXiv
  form. Next for the paper: DRAFT.md into LaTeX (docs/paper/paper.tex), compiled here.
- 2026-11-23 17:35 **arXiv asks for an endorsement in cs.NE.** Kris has the endorsement code (kept out of this
  public repository; in the assistant's private notes). Plan and a draft note in docs/paper/ENDORSEMENT.md: ask
  authors the paper cites, one or two at a time, only once the PDF exists and Kris has read it; Kris sends the
  notes. This puts the LaTeX version at the head of the paper's queue.
- 2026-11-23 18:00 **The paper compiles**: `docs/paper/build.py` turns DRAFT.md into paper.tex and paper.pdf (pandoc,
  pdflatex; both installed with pacman, with poppler to look at pages): 7 pages, figures 1 and 3 placed where the
  text points to them. DRAFT.md stays the text that is edited. The two built files and the author's name are kept
  out of the public repository for now (`AUTHOR="..." python docs/paper/build.py`). Kris confirmed the ORCID
  (found by a public search at Kris's request) and two earlier papers; one former coauthor has enough computer
  science papers on arXiv to endorse, for Kris to check with arXiv's own link and to ask (names kept out of this
  repository). Kris is linking ORCID and Zenodo and will reserve a DOI without publishing.
  Lives `plat_s0..2` at 40,000 / 40,000 / 35,000, all still in first words and rising (0.43 / 0.46 / 0.44 above the
  floor; under the old mark they would have left at 15,000–20,000 with 0.27–0.29). The lateral network's seed 2 is
  coded; its stream runs.
  Next: the draft brought up to date (no pass mark, "pages", the keyed store, the lateral network's three seeds,
  world 5), the [check] marks worked off against REFERENCES.md, figure 2 when a life has left a stage.
- 2026-11-23 18:02 **The paper rewritten for readers** (Kris: dull title, cold clipped writing). Three independent
  readers (style survey, hostile referee, cold reader); the referee's nine points checked against the logs, all
  held, and the draft now says them (VISION.md 18:15; queue D above). Working title "It Was the Code, Not the Rule";
  10 pages; both PDFs sent to Kris by mail at Kris's request. **Lateral network, three seeds**: +9.2 points without
  any relay, four fifths of the batch relay's 11.4. arXiv needs an endorsement in cs.NE (code in private notes;
  docs/paper/ENDORSEMENT.md); Kris's ORCID confirmed and linked to Zenodo; a former coauthor is the likeliest
  endorser. **Running: `organic-showcase`** (`experiments/showcase_probe.py` on out/cleanw_s0.pkl →
  out/tmp/showcase.log, showcase_cleanw_s0.json/.npz) for the section Kris asked for: what the learner does,
  with pictures, a timeline, its size and its answering times.
  Next: read the probe; the showcase figures and section; then queue D in order (16 first).
- 2026-11-23 18:08 **The showcase section** ("What it looks like", Kris's request): `experiments/showcase_probe.py`
  on out/cleanw_s0.pkl (300 + 300 held-out items, nothing learns) and two new figures from
  `experiments/paper_figures.py`: `fig_life` (one life: the two curves, the stages, what it babbled and named at
  four ages) and `fig_examples` (the first six test pictures and pages with its answers, not chosen). Measured:
  **the learner holds 257 million numbers** (gating policy 128 M, word cells and store 61 M, ear 51 M, cortex and
  relay 17 M, mouth 0.2 M); **answering time**: a question about a picture is answered after 1.0–1.6 ticks of
  looking, a spoken lesson with no picture after 3.1–4.1; right and wrong answers take the same time; under one
  eye move a question. So there are two speeds, and no evidence yet that the longer wait is thinking (one learner;
  three seeds owed). Paper now 12 pages; sent to Kris.
  Picture licences to check before anything is public: the pages in fig_examples may hold COCO or Commons pictures.
  Next: three seeds of the showcase probe; queue D (16: the end-to-end reference with a whitening layer).
- 2026-11-23 18:19 **Kris: the term "organic learning" and the title** "Organic Learning: Small-Scale Architecture
  Experiments Without Backpropagation" (set; the term defined in the introduction as five commitments; the phrase
  is the title of a US patent on network architectures, 10,846,592, to be cited). **Kris: is the judging lenient
  early and stricter later, and is there a sweet spot?** Answered and built (JUDGING.md, 2026-11-23): form is
  already judged more strictly with age, meaning strictly at every age; a near-miss level added to the judge
  (`NEAR`; "woof" for a dog 0.99) and `REWARD=strict|lenient|ramp` in the life script; two arms queued behind the
  plateau lives (`organic-reward-queue`), tests always strict. **Kris: answering speed, by training, organically**:
  the reward is what sets it (the gating policy's patience), so the reward arms are the first experiment on it.
  Next: read the plateau lives; queue D 16.
- 2026-11-23 18:30 **The like-for-like reference started** (queue D 16): `experiments/bench_backprop.py` has `RELAY=fixed`
  (the relay step settled on 6,000 unlabelled pictures seen through the filters as born, kept, errors going back
  through it) and `RELAY=tracked` (settled again every 5,000 pictures). The gradient was checked numerically in
  double precision, with and without the relay (largest relative difference 4e-4 on the filters, 1e-8 on the
  readout): the first reference's gradient is right too. Running: `organic-bench-bprelay`. Lives `plat_s0..2` at
  40,000, still in first words (0.43 / 0.46 / 0.49).
  Next: read the tuning (six runs of 10,000 pictures), then the three seeds; the lives' 45,000 and 50,000 tests;
  the showcase probe on seeds 1 and 2 when memory allows.
- 2026-11-23 19:25 **How often a thing comes again in a life** (queue D 19, first half; VISION.md 19:20): photographs
  0.1–1.9 times in 35,000 moments of first words, the adults' spoken lessons 5.6 times, social routines 66 times;
  the paper says so. **Backprop with the relay, tuning so far**: 0.597 at 10,000 pictures (0.52 without; word
  cells with the relay 0.623): the lead shrinks to a few points, as the referee expected; four tuning runs and six
  full runs to come. An explainer page for Kris (published as a private artifact; source docs/paper/explainer.html).
  PUBLISH.md brought up to date. Lives `plat_s0..2` at 45,000, still in first words (0.44 / 0.40 / 0.47; seed 1
  dipped). Memory: 3.5 GB free with the backprop queue beside the lives.
  Next: the tuning's end and the full runs; the lives' first stage change; the showcase probe on seeds 1 and 2
  when the backprop queue is done (memory).
- 2026-11-23 20:20 **The lives with no pass mark are still rising in first words** after 40,000–45,000 moments
  there: above the floor 0.49 (seed 0, 55,000 lived) / 0.46 / 0.50 (50,000 lived); the last three tests are above
  the three before by 0.04–0.05, the chance error of that difference is 0.03, so no stage has been left yet. The
  earlier lives were sent on after 5,000–10,000 moments at 0.27–0.29. At this pace a life of 120,000 moments ends
  about 06:00, and it may end still in first words: then the finding is the curve itself, and the pages stage needs
  longer lives. **Backprop with the relay, tuning**: fixed 0.597 / 0.572 / 0.527 at rates 0.001 / 0.003 / 0.01;
  tracked 0.602 / 0.564 (0.01 running). The best rate is the lowest tried, so a second queue
  (`organic-bench-bprelay2`) widens the grid after the first: a reference tuned at the edge of its grid would not
  be a fair one. **A drawing of the learner** for the paper (queue D 22): `fig_architecture`, each part filled by
  how it comes to be (local rule online / batch before learning / reward / fixed reflex); the paper has 13 pages.
  Memory 2.5 GB free: nothing heavier was started.
  Next: the backprop full runs (about 100 minutes each, six of them, then the wider grid); the lives' stage
  change or end; settings table, compute and the judge's instruction for the paper; the showcase probe on seeds 1
  and 2 and queue D 18 when memory allows.
- 2026-11-23 21:08 **The first life to leave a stage with no pass mark**: `plat_s1` left first words at 55,000
  lived (45,000 in the stage), level 0.45; its last three tests were above the three before by 0.025, under the
  chance error 0.030. Its last single test was its best (0.49), so this may be the chance leave foreseen at 20:20
  (the dip at 45,000 sits in the newer window). Seeds 0 and 2 are still there (0.49, 0.48; margins 0.049, 0.033).
  **The paper's appendix** (queue D 22, now done): settings table, compute, the judge's instruction word for word.
  Collecting the settings turned up something the draft had never said: **each word-cell synapse's step shrinks
  with its own use**, 1/(1 + u/5) with u the running sum of that input's squared activity (`maturation` in
  organic/words.py): an AdaGrad-like step. It is local, and now stated in the paper's description of the word
  cells; it bears on the reading of "the rule is a delta rule" and of the relay (a per-input step already evens
  out inputs of different size, which the relay then does for their correlations). Two wrong values caught while
  writing the table (ticks at most 12, not 6; store 2,000, the 6,000 is only with replay). **A layout fault
  fixed**: a page with a table and floating figures ran past its bottom margin and lost text (pandoc's longtable
  does not see floats); docs/paper/build.py now turns the tables into ordinary ones; 14 pages, no overflow.
  Backprop with the relay: tuning done (tracked 0.01: 0.533), full runs started with rate 0.001.
  Next: the full runs; `plat_s1` in the pages stage, the other two leaving; queue D 17–18 when memory allows.
- 2026-11-23 21:25 **Kris read draft 5 and gave directions** (rule 8 and queue E above): improve only organically,
  and adopt what is well established in biology also when it moves no number; a learned gaze; the cortex's sizes
  and architecture from biology and recent work; recent papers; the learner stays on the processor (the graphics
  card is full with the two adult models, and a moment's time is spread over many small steps: CRAFT.md,
  2026-11-18). Paper: the patent note moved to the end, the world-reverting bug taken out of the mistakes (Kris:
  bugs are not research mistakes; it touched no life the paper reports), the judge told as the wrong assumption
  it was, the repository's link, where the sizes come from. Three surveys run in the background (gaze; early
  visual cortex in numbers and recent local-rule vision models; papers of 2022–2026).
  What the code does today, read for the gaze: "looking" is not a sequence of eye movements. In one glimpse the
  word cells get the whole view and up to four fixated windows at once, and each cell takes the view that drives
  it most. So there is no order of looking, no cost of looking and no choice to learn. A learned gaze starts by
  making looking serial: one fixation a tick, the next place chosen.
  Next: read the three surveys; the organic check for a serial, learned gaze; then the probe.
- 2026-11-23 21:27 **The survey of recent work is back** (docs/RESEARCH_RECENT.md; queue E 25). **The paper's main
  finding is known from the other side**: Thiry et al. 2021 (whitened patches and a linear classifier match
  trained networks on CIFAR-10; training the patches changes nothing), Prabhu et al. 2024 (random features,
  decorrelation and class means in one pass match continually trained networks; the decorrelation matters most),
  Dalm et al. 2024 (decorrelated inputs speed up backprop itself). The draft now says so in the introduction and
  cites 26 recent papers in related work (all in REFERENCES.md, none opened by a person yet); 15 pages. What stays
  ours: the finding reached from a learner built of local rules, the like-for-like controls in a one-pass stream,
  the staged life. **"Hebbian is no better than random" is scoped to hard winner-take-all**: SoftHebb's own
  control says the same for hard WTA and reports a gain for soft WTA, which we never built faithfully.
  **Owed from this** (queue E): 27. a faithful soft winner-take-all development of the filters (the published
  rule, unchanged), measured as the developed filters were; 28. the relay by interneuron gains (Duong et al.
  2023), a published local and online whitening, beside the lateral network.
  Next: the two other surveys (gaze; early visual cortex), then the organic checks.
- 2026-11-23 21:45 **The gaze and cortex surveys are back** (docs/RESEARCH_GAZE.md, docs/RESEARCH_CORTEX.md) and
  the plan with its organic checks is in VISION.md (21:45). Gaze: make looking serial (one fixation a tick, the
  periphery coarse), let the heard word's cells bias the priority map (no new rule), a three-factor value rule
  on the choice of place, staying by the patience that exists. Cortex, by how established and how far off we
  are: divisive normalisation, complex cells and stages, several patch sizes, a retina in place of batch
  whitening, soft competition, filters before experience, no shared weights.
  Next: `experiments/look3_probe.py` (the serial gaze as a probe on pages with a saved learner; memory: check
  look2_probe's need first); the backprop full runs; the lives.
- 2026-11-23 22:02 **Backprop with the relay, the whole stream, seed 0: 0.682** (fixed relay, rate 0.001;
  out/bench/backprop_relay_full.log). Without the relay the same seed had 0.627; the word cells with the relay
  have 0.687 ± 0.005 over three seeds. One seed, but it says what the literature says: given the same
  decorrelated code, end-to-end backprop is level with the local learner. The paper's pending mark carries it.
  Lives at 60,000: `plat_s1` in pages (0.17 above the floor on its first test there; first words 0.45);
  seeds 0 and 2 still in first words (0.46, 0.49).
  **The first gaze probe is written and running**: `experiments/look3_probe.py`, unit `organic-look3` (capped
  at 1 GB; seeds 0–2, 6,000 moments each, about 15 minutes a seed) → out/tmp/look3.log. The word cells are
  raised as today (the whole view and all fixated places at once); then, nothing learning, the looks are taken
  one at a time in three orders: the reflex's, the heard word's (the place whose coarse view most drives the
  cells of the words just heard), random. Read: on which look the named thing is in the fovea; the answer with
  the first one or two looks.
  Next: read the probe; if the word leads the eye, the serial gaze goes into the learner behind a switch; the
  backprop seeds 1 and 2.
- 2026-11-23 22:29 **The heard word leads the eye** (VISION.md 22:29; two seeds, the third running): with the
  looks taken one at a time, the named thing is in the fovea on the first look on 0.82 / 0.85 of pages when the
  places are ordered by how much their blurred view drives the cells of the words just heard, against 0.46 / 0.46
  in the reflex's order and 0.38 / 0.41 at random. No new rule: the word cells' own synapses. Found in the code:
  the learner has two eyes; the gating policy already learns eye commands by reward on the old one, and the word
  cells are served by the new one, which does not move. Joining them is the step.
  Running: `organic-look3b` (after `organic-look3`): how sure the word cells are look by look, stopping by a
  threshold, how sure before and after the named thing enters the fovea → out/tmp/look3b.log.
  Next: read look3b; then serial looking in `organic/speech.py` behind a switch that leaves today's behaviour
  untouched by default (the reward lives start by themselves when the plateau lives end and must stay comparable).
- 2026-11-23 23:45 **Both gaze read-outs are in, three seeds** (out/tmp/look3.log, look3b.log; VISION.md 23:45).
  The heard word's order puts the named thing in the fovea on the first look on 0.82 / 0.85 / 0.81 of the pages
  where it matters (reflex 0.46, random 0.38–0.41), and answers right with one look on 0.67 / 0.70 / 0.71
  (reflex 0.60 / 0.61 / 0.63). **When to stop is the weak part**: how sure the word cells are rises when the
  named thing enters the fovea (0.39 → 0.43, 0.37 → 0.43, 0.42 → 0.46) but separates right from wrong answers
  only a little (0.44 against 0.39), and for "how many" hardly at all (0.34 against 0.33); stopping at a level of
  sureness saves almost no looks in any order. Counting gains nothing from the order: every place must be seen.
  **The serial gaze is in the learner, behind a switch** (`organic/speech.py`, `gaze = "serial"`; life env
  `GAZE=serial`): the word cells first get the whole view only; each eye command of the gating policy that lands
  on a place that stands apart brings them what the fovea has there; the map the commands choose on is biased by
  what stands out, by the heard words' pull and against places already seen; the adult's reply (or a comment)
  draws the eye once before the word cells learn. The eye commands are the gating policy's, credited by reward as
  they always were (eligibility × reward prediction error: a three-factor rule already). Default behaviour is
  untouched: the same digest of 300 moments before and after the change.
  **Running: `organic-gaze1`** (the small learner, both ways, three seeds). Not possible yet: the same from a
  grown learner's checkpoint (memory: 3.3 GB free beside three lives, a checkpoint needs about 1.6).
  Backprop with the relay, seed 1: **0.690** (seed 0: 0.682); the paper's pending mark carries both.
  Lives: `plat_s2` left first words at 60,000 lived (exact grade 0.54); `plat_s0` is still in them at 70,000
  (0.54–0.56); `plat_s1` in pages, exact grade 0.43 at 65,000, `plat_s2` 0.42 on its first test there.
  (Corrected 2026-11-24: these are the log line's exact grades, which I had called "above the floor".)
  Next: read gaze1; the backprop third seed and the tracked relay; queue E 24 (the first cortex probe: divisive
  normalisation) when the gaze probe has freed its memory.
- 2026-11-23 23:48 **The first cortex probe is written and queued** (queue E 24, item 1; `organic-norm1`, after
  the gaze probe): divisive normalisation in place of "above the mean, rectified". R = D² / (σ² + pool), D the
  rectified drive times the window's contrast, the pool all kinds at the place (and, in a second arm, the
  surround), σ² the median pooled energy on unlabelled pictures: nothing searched. Read first for what it does
  to the code (sparseness over kinds and over places, response against contrast), then the stream. Under rule 8
  the stream's number does not decide adoption. The gaze probe was restarted with seed 0 alone so that this can
  follow it sooner.
  Next: read gaze1 (seed 0, both ways) and decide on seeds 1–2; read norm1.
- 2026-11-24 00:58 **The serial gaze as first built is never used** (VISION.md 00:58; gaze1, seed 0): the small
  learner answers at about the third tick, before its first eye command can be given, so it answers pages from
  the coarse view: pages 0.44 against 0.56 with the lives' looking at 6,000 moments, "is there" 0.55 against 0.67;
  looks before answering 0.01. And in both arms the eye commands die out by 3,000 moments: a gaze taught by
  reward and the cost of time alone learns not to look. The build had the reply draw the eye by reflex and not
  the question, which is not what is known (a heard word draws the eye within 200 ms, before any answer).
  Changed: `look_on_question` (the question draws the eye once, one tick). `organic-gaze2` reruns the serial arm
  after checking the default path's digest; `organic-norm1` waits behind it.
  **Backprop with the relay, three seeds: 0.688 ± 0.005** (0.682, 0.690, 0.691) against 0.687 ± 0.005 for the
  word cells: level. In the paper, with the wider tuning grid and the tracked relay still pending.
  Lives at 70,000–75,000: `plat_s0` still in first words (exact grade 0.52 at 75,000); pages `plat_s1` 0.47,
  `plat_s2` 0.45 (exact grades; corrected 2026-11-24, I had written "above the floor").
  Next: read gaze2 and the digest; seeds 1–2 of the three arms if seed 0 says something; read norm1.
- 2026-11-24 10:13 **The loop did not wake between 01:42 and 10:08** (Kris asked for the status at 10:08; nothing
  was read in between; the units ran by themselves). What was found:
  **The plateau lives ended as planned at 120,000 moments** (the health check's "stopped without a verdict" is
  how a life that reaches TOTAL looks; none was killed). All three left every stage with no pass mark: first
  words after 70,000 / 45,000 / 50,000 moments there, pages after 30,000 / 35,000 / 40,000, and ended in
  preschool. Above the floor (the records' `above`; the three numbers first written here were exact grades,
  corrected the same morning): first words when left 0.53 / 0.49 / 0.49; pages when left 0.33 / 0.31 / 0.36;
  **preschool never rose above blind answering** (0.00 / 0.02 / 0.02 after 10,000 / 30,000 / 20,000 moments
  there), and while the learner was there first words fell to 0.42 / 0.39 / 0.39 and pages to 0.27 / 0.25 / 0.30. **The reward lives started by themselves** (`kind_s0..2`, in the babbling stage).
  **Backprop with the relay settled again along the stream: 0.671 / 0.689 / 0.693** (mean 0.684); the wider
  tuning grid found nothing better than rate 0.001, batch 32 for either relay. The comparison is closed: 0.688
  ± 0.005 (fixed relay) against 0.687 ± 0.005 for the word cells.
  **Serial gaze with the question drawing the eye** (gaze2, seed 0): the learner now looks before it answers
  (0.87–0.93 looks), pages 0.50 at 6,000 moments (all at once 0.56; without the question's look 0.44), "is
  there" 0.62 (0.67; 0.55). **But the first look is on the named thing on only 0.46–0.51 of pages: the reflex's
  level, not the 0.82 of the read-out.** Suspected: the pull's size against the reflex's fixed half (the
  read-out ordered by pull alone); to be measured, not guessed.
  **The default path is unchanged**: the digest check as queued compared against a copy of the old code that
  could not find the adults' wordings (data/ is found beside the code), so its two lines differed; with the
  data linked both give c52819a8c3a93a7b.
  **`organic-norm1` was killed for memory** (settling the relay needs more than its 1.3 GB cap); restarted at
  10:12 with 2.6 GB (4.6 GB are free: the new lives are young).
  Next: the pull's size in the serial gaze; read norm1; the plateau lives into JUDGING.md and the paper.
- 2026-11-24 10:34 **Kris: "why fixed, research a better solution … did we find more interesting sota research
  that we can take architecture ideas from".** Two surveys run and written: how the map combines its sources
  (RESEARCH_GAZE.md, second part) and architecture ideas of 2023–2026 (RESEARCH_ARCH.md; queue E 29). The fixed
  half is gone: each source's places compete, the map is their weighted average, the weights are learned from
  reward on a fixed total, born with the reflex alone (VISION.md 10:34; queue E 30). Default digest unchanged.
  `organic-gaze3` running. Divisive normalisation, the code itself (norm_probe.log, seed 0): sparseness over the
  kinds at a place 0.83 against 0.71 now, kinds above half the strongest 3.9% against 8.5%, over places for a
  kind 0.87 (0.90 with the surround) against 0.72; the response at a quarter and half the contrast is 0.13 and
  0.41 of that at full (now: 1.00 and 1.00, no dependence on contrast at all). The streams are running.
  Next: read gaze3 and the streams; plateau lives into JUDGING.md; then queue E 29a (the temporal term).
- 2026-11-24 10:36 **Rule 9 from Kris** (above): iterate on the architecture in the loop, organic over
  performance, measured beside the current paradigms; drafts by email. **Draft 6 sent** to Kris (15 pages: the
  backprop comparison closed, the plateau lives, without the gaze and cortex work, which have one seed).
  The plateau lives are written into JUDGING.md (2026-11-24).
  Next: read gaze3 and the three streams of the cortex probe; then queue E 29a.
- 2026-11-24 11:16 **The word's weight in the eye's map grows from reward** (gaze3, seed 0; VISION.md same time): born 0,
  it is 0.15 / 0.46 / 0.58 / 0.75 after 1,000–4,000 moments of pages; the first look lands on the named thing on
  0.65 of pages (reflex 0.49); with one look before answering the serial learner is level with the one handed
  every place (pages 0.53 against 0.56, "is there" 0.66 against 0.67). One seed. `organic-gaze4`: a control with
  the pull shuffled over the places, then seeds 1 and 2.
  **Divisive normalisation on the stream, seed 0**: level with the present response (word cells with the relay
  0.624 against 0.613; ridge 0.654 against 0.659), the code sparser and saturating with contrast; the surround
  pool a little weaker (0.610). Seeds 1 and 2 queued behind the gaze runs.
  The health check no longer calls a life that ran to its planned end a death (out/monitor/ended.json).
  Next: queue E 29a, the temporal term: organic check and design while the probes run.
- 2026-11-24 11:33 **The drift probe is written, smoke-tested and queued** (`organic-trace1`, behind `organic-gaze4`;
  VISION.md 11:28). Smoke test on 40 fixations only (says nothing yet): answers to a picture and to it two
  pixels aside alike 0.854 in the first stage, 0.773 (plain) and 0.794 (trace) in the second.
  The shuffled-pull control so far: the pull's weight 0.01 and 0.00 at 2,000 and 3,000 moments (the real arm:
  0.01 and 0.15).
  Next: read gaze4 (control, seeds 1–2), then the normalisation seeds, then trace1.
- 2026-11-24 12:28 **The shuffled-pull control fails the first rule for the map's weights**: with a pull that says nothing
  its weight still rose to 0.53 by 6,000 moments (real: 0.75). The claim of 11:16 that the word's weight "grows
  from reward" is withdrawn (VISION.md, same time). The rule had a selection bias (the highest place always taken,
  eligibility against the plain mean), a floor that noise climbs, and a rate far too fast. Corrected: the look
  drawn by priority while living, eligibility against the draw's own average, the slow rate of a stable value
  (0.001). `organic-gaze4` and the first rule's seeds stopped; `organic-gaze5` (slow pair), then `organic-trace1`
  (the drift probe), then `organic-rest1` (fast pair, seed 2's baseline, normalisation seeds 1–2).
  Default digest unchanged (c52819a8c3a93a7b).
  Next: read gaze5; the drift probe.
- 2026-11-24 13:28 **The corrected rule at the slow rate, real arm** (gaze5_slow.log, seed 0): the pull's weight is 0.000 /
  0.001 / 0.005 / 0.008 / 0.010 at 4,000–12,000 moments: slow and steady, about what was estimated; the first look
  stays the reflex's (0.49–0.51); pages 0.59 and "is there" 0.75 at 12,000 with 0.86 looks before answering. The
  shuffled control is running; the reading waits for it. Reward lives at 25,000: all three in first words
  (exact grades 0.43–0.46).
  Next: the control; then the drift probe starts by itself.
- 2026-11-24 14:30 **Reward does not teach the map's balance** (VISION.md, same time): corrected rule, slow rate, 12,000
  moments: the pull's weight 0.010 real against 0.006 shuffled; first look at the reflex's level in both. Negative,
  kept with its control. **The balance is now set by its time course** (`gaze_balance = "time"`): what stands out
  is a transient (gone a quarter of a second after the scene appeared), the heard words' pull stays; nothing
  learned, no weight of mine. Default digest unchanged. Running: `organic-gaze6`; `organic-trace1` (the drift
  probe, first stage read: answers to a picture and to it two pixels aside alike 0.847); `organic-rest1` behind.
  Next: read gaze6; three seeds of serial (time) against all at once; the drift probe.
- 2026-11-24 15:18 **The balance by its time course, seed 0** (gaze6; VISION.md same time): the first look goes to the named
  thing on 0.51 → 0.70 of pages as the word cells learn (reflex 0.49), nothing learned about the balance; pages
  0.50 against 0.56 all at once, "is there" 0.68 against 0.67, with 0.86 looks before answering. Seeds 1–2 queued
  behind the drift probe; the fast learned-weights pair dropped (settled). Drift probe: plain arm done (second
  stage 0.799 alike under a two-pixel shift, first stage 0.847); trace arm developing.
  Next: read trace1 (trace arm, streams); then rest1.
- 2026-11-24 16:18 **The drift probe, seed 0** (VISION.md same time): the trace raises the second stage's tolerance to a
  two-pixel shift (0.826 against 0.799 without it; the first stage's pooled output 0.847); on the stream the
  second stage helps the word cells (plain 0.646, trace 0.631, first stage alone 0.613) and not the ridge (0.655,
  0.640, 0.659). Carried forward on its ground with its cost stated; seeds 1–2 queued (`organic-trace2`).
  `organic-rest1` is on the gaze seeds.
  Next: read rest1 (gaze seeds 1–2, seed 2's baseline, normalisation seeds); then the paper's new section when
  three seeds are in; then queue E 29b (a blurred start).
- 2026-11-24 16:48 Kris: "ok keep going keep on looping and improving". **Queue E 29b written and running** (`organic-diet1`;
  VISION.md same time): the filters develop on a maturing eye's view (waves, then pictures sharpening and gaining
  colour); read on the filters themselves, the stream, and the stream with blurred test pictures for both cortices.
  Next: read rest1 (gaze seeds), diet1; then trace2 runs.
- 2026-11-24 17:31 **The serial gaze is level with all at once on three seeds** (VISION.md same time): pages 0.57 against 0.55,
  "is there" 0.71 against 0.70, with 0.87 looks before answering; first look on the named thing 0.68–0.70.
  **A blurred start, seed 0: nothing moved** (0.616 against 0.613 sharp, 0.585 against 0.588 blurred); kept as
  measured, seeds 1–2 running. Kris asked how to keep the loop running without the laptop: answered (tmux on the
  desktop and Remote Control; "Keeping the loop alive" above).
  Next: rest1 (normalisation seeds), trace2, diet2; then the paper's section on looking and on the cortex from
  biology, and draft 7 by email.
- 2026-11-24 17:40 The loop runs inside herdr on the desktop now (Kris moved the session; the process no longer hangs on an
  SSH connection). **The paper has a section on looking one place at a time** (three seeds of the small learner,
  the read-out, the timing account) and a fourth mistaken assumption (that reward could teach the eye what to
  trust; the shuffled control). Six references added to REFERENCES.md, unopened. Normalisation: seed 1 coded
  (sparseness 0.825 against 0.711, as seed 0), streams running; blurred start seed 1 developed (filters 0.281
  against 0.297 cycles a pixel, colour 0.593 against 0.558).
  Next: the streams of seeds 1–2 (normalisation, blurred start), trace2; then the paper's section on the cortex
  from biology and draft 7 by email.
- 2026-11-24 18:46 All healthy, 4.0 GB free; streams of seeds 1–2 still running (normalisation seed 1 repeats seed 0: word
  cells with the relay 0.636 against 0.616, without 0.529 against 0.522, the ridge without the relay 0.637
  against 0.644; blurred start, seeds 1–2: filters again coarser and more coloured, 0.281 and 0.275 against 0.297
  and 0.295 cycles a pixel). Reward lives at 50,000 moments: all three still in first words (exact grades
  0.48 / 0.52 / 0.53). **Queue E 29f written and queued** (`organic-expand1`; organic check in VISION.md, same
  time): a wide sparse layer between the relay and the word cells, sized from the mushroom body and the dentate
  gyrus; smoke test on 1,000 pictures: the sparse layer loses more than its width gives back (says little yet).
  PUBLISH.md has today's state. The Running section above was brought up to date.
  Next: read the normalisation and blurred-start seeds, write the paper's section on the cortex from biology
  (the drift stage marked as one seed until trace2 is in), email draft 7; read expand1; then the organic check
  for queue E 29d (surprise sets how much is learned).
- 2026-11-24 19:37 **Divisive normalisation and the blurred start on three seeds** (VISION.md, same time; 10,000 pictures).
  Normalisation: the code changes as the physiology says on every seed (sparseness 0.71 → 0.83; a response
  that grows with contrast); word cells with the relay 0.630 against 0.620, the ridge 0.650 against 0.664 (down
  on all three); the surround pool adds nothing. Adopted for the next cortex on its ground, nothing claimed.
  Blurred start: filters coarser and more coloured every time, the stream level sharp and blurred (0.620 and
  0.620; 0.580 and 0.583). **The sparse layer, two seeds** (expand_probe_s{0,1}.log): two pictures half as alike,
  three times the directions; naming new pictures falls (0.61 → 0.57 at ten times the cells, 0.51 with as many
  cells) and a picture seen once is remembered better (0.88 → 0.95); no inhibition: no change. It belongs in
  front of the store, not the word cells. A first test of it as the store's key
  (`experiments/expand_store_probe.py`, three seeds) is too easy to tell keys apart, and shows the sparse layer
  does not replace the store's hand-set steepness. **The paper has the section "Giving the cortex what a cortex
  has"** (17 pages; seven references added, unopened; the sparse layer on two seeds and the drift stage on one,
  marked). Trace2: seed 1's first stage read (0.849 alike under a two-pixel shift).
  Next: seed 2 of the sparse layer and the wiring with 100 inputs, then draft 7 by email; trace2; the organic
  check for queue E 29d.
- 2026-11-24 20:13 **The sparse layer on three seeds** (VISION.md, same time): with seven inputs a cell, naming new pictures
  0.578 against 0.620 at ten times the cells and a picture seen once 0.95 against 0.85. **With a hundred inputs
  a cell (seed 0) the cost on naming is gone** (0.612 against 0.613 at ten times the cells; seen once 0.98;
  slower at 500 pictures): the reading "not in front of the word cells" is taken back to undecided; seeds 1–2
  queued (`organic-expand2`). Paper's paragraph rewritten to say so. **Queue E 29d designed and queued**
  (`organic-rpe1`; organic check in VISION.md): of "surprise sets how much is learned" only one part was not
  already settled: the gating policy's reward error is now, behind a switch, taken against its own usual size
  (adaptive coding by dopamine neurons, in the form of divisive normalisation); default digest unchanged; the
  fixed-gain control runs with it. Trace2: seed 1's plain arm developed (0.811 alike under a two-pixel shift;
  first stage 0.849).
  Next: expand2's two seeds into the paper, then draft 7 by email; rpe1; trace2.
- 2026-11-24 21:06 **The sparse layer with a hundred inputs a cell, three seeds** (VISION.md, same time): at ten times the
  cells the word cells name new pictures at 0.609 against 0.620 and remember a picture seen once 0.975 against
  0.849; slower at 500 pictures (0.26 against 0.39); the ridge 0.616 against 0.664. Decided: not put into the
  learner (a life's words are the stream's first 500 pictures; the store is the learner's memory of moments);
  kept as measured; no further wirings (that would be a search). **Draft 7 (17 pages) emailed to Kris.**
  Running: `organic-rpe1` (seed 0, first arm), `organic-trace2` (seed 1: trace arm 0.837 alike under a
  two-pixel shift, plain 0.811, first stage 0.849; streams next), the reward lives.
  Next: read rpe1 and trace2 as they come; reward lives at 75,000; then the serial gaze from a grown learner's
  checkpoint when 1.6 GB is free; queue E 29e (what comes back in sleep is tagged at the moment): organic check.
- 2026-11-24 22:14 All healthy, 2.5 GB free. **Read:** the drift stage on seed 1 (the trace gives back the tolerance a plain
  second stage loses, 0.837 against 0.811, first stage 0.849; on the stream the two do not differ: 0.626 and
  0.624, ridge 0.639 and 0.635); the reward's error against its usual size on seed 0 (the normalised error stays
  near 0.45 while the raw one grows from 0.22 to 0.37, so the step falls by itself from 1.9 to 1.3 times the raw
  one; grades as the baseline and the fixed-gain control within one seed's noise); the reward lives at 65,000
  (two in pages, as the plateau lives at the same ages: no difference from a kinder reward so far; JUDGING.md).
  **Queue E 29e: organic check written, built behind a switch, test queued** (VISION.md, same time): a stored
  moment keeps the reward that came with it and only such moments return in sleep (`Episodes.tag`, life env
  `REPLAY_TAG=reward`; default digest unchanged). From the plateau lives' records it would make the earlier
  stages about four parts in ten of what returns (one in ten when replay was tried) and remove talk from sleep
  by itself. The test is at life scale, in preschool where the loss is real (`organic-tag1`, waits for memory
  behind the reward lives). Expectation written down: a small effect at most.
  Next: rpe1 seeds 1–2 and trace2 seed 2 as they come (then the paper's drift row, three seeds); reward lives at
  75,000; while tag1 waits: queue E 26/28 (the relay by local rules: the lateral network among the summary's
  numbers, interneuron gains), which needs little memory; the serial gaze from a grown learner's checkpoint
  when memory allows.
- 2026-11-24 23:11 All healthy, 2.2–2.4 GB free (at the floor: nothing heavy started). Read: the reward's error on seed 1
  (as the baseline; control for it running); the drift stage's seed 2 is in its trace arm; the reward lives at
  70,000 (two in pages, one in first words). **Queue E 26/28: organic check, probe written and queued**
  (VISION.md, same time): the relay learned online by a published interneuron circuit (Lipshutz et al. 2022 with
  the leak of Duong et al. 2023b; equations read from the papers tonight). The cell's leak and the fact that
  inhibition only takes away put the floor at the mean variance, which measures 0.00094 of the largest where my
  hand-set `shrink` was 0.0005: the constant is not needed. The gain-only circuit (28 as written) would need 8.4
  million interneurons: not built. `organic-relay1` runs behind trace2.
  Next: read the smoke test and seed 0 of relay1 (then seeds 1–2, and the pictures of development coded so the
  developed arm does not use the stream's own); rpe1 seeds; trace2 seed 2 into the paper's drift row; reward
  lives at 75,000; tag1 waits for memory.

- 2026-11-25 00:26 All healthy, 2.7–3.5 GB free. **Read:** the drift stage on three seeds (the trace gives back half of the
  tolerance a second stage loses, 0.805 → 0.829 of 0.848, and changes nothing on the stream; the paper's row and
  paragraph are filled, its pending mark gone); the reward's error on three seeds with its control (grades as
  the baseline; **adopted for lives born from now on**, `RPE=normalised` in the launch line; the eye stops
  moving on "is there" pages with either larger step, to be watched with the serial gaze); the reward lives at
  75,000 (all three in pages, no difference from the plateau lives; JUDGING.md). **The relay circuit, first
  run:** its resting point is as good as the batch relay (0.609 against 0.613, with no constant of mine) but at
  the papers' rate the circuit does not reach it at this size (0.584, and further from rest after 10,000
  pictures than after 1,000); worked out why (one picture's step is rate × |y|², tens of per cent with 4,096
  cells) and checked on a small made-up input; the full-size reading is queued (`organic-relay2`). The
  development pictures are being coded so no arm uses the stream's own. **Queue E 27: organic check, probe
  written, smoke-tested and queued** (`organic-sh1`): the published SoftHebb layer exactly as in the authors'
  code, developed against as born, in our measure and, as the control that it is their rule, in theirs in
  small.
  Next: read relay2 (then fix the rate's meaning, run the arms on three seeds on the relay's own pictures);
  read sh1 seed 0 (then seeds 1–2, or the paper's sentence); reward lives at 85,000; tag1 waits for memory;
  email draft 8 when the relay or the soft rule changes what the paper says.
- 2026-11-25 01:15 All healthy, 4.3 GB free. **Read:** the relay circuit direction by direction at full size: at the papers'
  rate one picture's step is 0.38 and the circuit stops 10–40% above the floor and wanders; at a tenth of it
  (step 0.03) it goes on approaching its resting point over 20,000 pictures (strong directions within 13%).
  The rule is kept unchanged with the rate said in this size's units; no scaling mechanism of mine added. The
  six arms run on three seeds on the batch relay's own unlabelled pictures (`organic-relay3`). **The published
  soft rule, seed 0:** in its authors' measure in small the developed layer names 0.616 against 0.515 as born:
  it replicates, and my stated expectation was too low; the summaries of two pictures are half as alike and
  use three times the directions. Its reading in our measure runs (`organic-sh2`; the first queue stopped on a
  file fault of mine, mended); seeds 1–2 queued (`organic-sh3`). PUBLISH.md brought up to date.
  Next: read sh2 (our measure); if the soft rule is ahead there too, rewrite the paper's section and write the
  organic check for the rule on our decorrelated windows; relay3 as seeds come; reward lives at 85,000.
- 2026-11-25 02:23 All healthy, 3.4–3.7 GB free. **The published soft rule in our measure, seed 0: word cells 0.689 at
  10,000 pictures against 0.549 for the same layer as born and 0.613 for our cortex (ridge 0.700 / 0.603 /
  0.659). My stated expectation (under 0.02) was wrong.** The paper's paragraph carries the seed-0 numbers with a
  pending mark; seeds 1–2 started at once (`organic-sh4`); the dissection (hard / no unlearning / no layer-wide
  division) is queued on seed 0 (`organic-sh5`). **The relay circuit at a step of 0.03, seed 0:** every arm
  within 0.013 of the batch relay; developed on the relay's pictures and then learning on along the stream:
  0.613, the batch relay's own number. Seeds 1–2 running (`organic-relay3`). Reward lives at 85,000: nothing new.
  Next: sh4 and relay3 as seeds come; then, if three seeds agree: the paper's section, abstract and the
  like-for-like baselines redone on the soft layer (streaming LDA on its code at once; backprop of its shape
  needs `bench_backprop.py` given the layer's shape); the dissection; then the rule in our cortex's place (the
  eye's 64 × 64 pictures, decorrelated windows) after its organic check. Draft 8 by email when the three seeds
  are in.
- 2026-11-25 03:27 All healthy, 3.5 GB free. **The soft rule, seed 1, says what seed 0 said:** word cells + relay 0.684 at
  10,000 pictures against 0.546 for the same layer as born (seed 0: 0.689 / 0.549); ridge 0.700 / 0.607; their
  measure in small 0.613 / 0.515. Seed 2 is in its streams. **The relay circuit, seed 1:** batch 0.616, resting
  point 0.601, developed and held 0.590 / 0.598, from nothing 0.606, developed then learning on 0.608 (0.008
  under the batch relay; on seed 0 equal). Seed 2 running. Reward lives at 90,000: nothing new. **Written and
  checked: the references for the soft layer** (`bench_soft_codes.py`: the whole stream coded;
  `bench_backprop_soft.py`: backprop of the layer's shape, its way in equal to the probe's codes and its way
  back checked against small changes), queued as `organic-softbase1` behind relay3, sh4 and sh5 (about eight
  hours: tuning on seed 0, then three seeds on the whole stream). Next: seed 2 (about 04:05): the paper's
  paragraph with three seeds at 10,000 pictures, the abstract's sentence on developed filters made a statement
  about our hard rule, draft 8 by email; relay3 seed 2 and the decision on the batch relay; the dissection (sh5);
  then softbase1's numbers into the paper's comparison with backprop as they come (read
  backprop_soft_check.log first).
- 2026-11-25 04:11 All healthy, 3.4 GB free. **Three seeds, the soft rule:** word cells + relay 0.689 ± 0.005 developed,
  0.548 ± 0.002 as born, 0.620 ± 0.010 our cortex (ridge 0.700 / 0.605 / 0.664). "Developed filters do no better
  than random" is about our hard rule. Nothing adopted before the dissection (its hard arm already shows the
  near-copy collapse ours had: 916 pairs above 0.9). **Three seeds, the relay circuit:** developed then learning
  on 0.614 ± 0.007 against the batch relay's 0.620 ± 0.010 (0.000 / 0.008 / 0.010 under). **Decision (rule 8):**
  the circuit is the relay's form for the next lives if a life-scale test shows no harm; first the circuit in
  `organic/cortex2.py` behind a switch (digest c52819a8c3a93a7b must hold), then a probe on a grown learner's
  level, then lives. **Paper:** abstract and two sections updated, Lipshutz 2022 added unopened, **draft 8 (18
  pages) emailed to Kris.** Next: the dissection's other arms and streams (sh5, about 05:10); the relay circuit
  behind a switch in the cortex; softbase1 as it comes (backprop_soft_check.log first); then the organic check
  for the soft rule in our cortex's place.
- 2026-11-25 04:20 All healthy, 3.5 GB free. **The relay circuit is in the cortex's code** (`organic.cortex2.RelayCircuit`,
  a drop-in for `Relay`; one hook in speech.py so that it does not learn from a test; the batch relay
  untouched). Default path unchanged, checked side by side (committed and edited code both 2baf9589ac5900bc).
  **The digest check itself:** its absolute value drifts (c52819a8c3a93a7b on 2026-11-24) because the world's
  files grow; from now on the check is committed against working code at the same time
  (`git archive HEAD organic | tar -x -C out/tmp/head_root`, `ln -s $PWD/data out/tmp/head_root/data`,
  `ROOT=out/tmp/head_root`). The same growth means each generation of lives starts in a slightly larger world
  (put to Kris above). `organic-mkcircuit1` develops the circuit for a life's cortex →
  data/visual_cortex2_circuit.pkl, log out/tmp/make_circuit.log (MemoryMax 1300M; if it was killed for memory,
  run it again with 1800M when softbase1 is not at its peak). Next: sh5's arms and streams (about 05:10); the
  circuit's reading in make_circuit.log; softbase1 as it comes (backprop_soft_check.log first); the organic
  check for the soft rule in our cortex's place; a life-scale test of the circuit when memory allows.
- 2026-11-25 05:07 All healthy, 3.5–3.9 GB free. **The dissection (seed 0):** hard 0.512 and soft-without-unlearning 0.444
  against published 0.689 and as born 0.549: the losers' unlearning does the work (without it the kinds pile
  onto the same windows, as ours did); with the layer-wide factor held at its median the weights ran away.
  Organic check part by part in VISION.md: everything is local but the division by the largest change in the
  layer. **`organic-sh6`** (out/tmp/softhebb_dissect2_queue.sh, MemoryMax 1500M, about 45 minutes): two more
  arms, own (each kind divides by its own largest change) and early (the factor held at its early value) →
  softhebb_dissect2_s0.log, stream_{sho,she}10k_s0.log. **The circuit for a life** is made
  (data/visual_cortex2_circuit.pkl; 208 directions against the batch relay's 271 on 500 lived pictures).
  `organic-softbase1` started 04:45: seed 0's whole stream coded (equal to the kept codes), seeds 1–2 in work.
  Reward lives at 95,000: seeds 0 and 2 in preschool. Next: sh6; softbase1 (backprop_soft_check.log first, then
  the tuning); if own holds: the soft rule in our cortex's place (the eye's 64 × 64 pictures; raw standardised
  windows and decorrelated windows as two arms) after reading BCM / ABS in the papers; the dissection on seeds
  1–2 for whatever goes into the paper.
- 2026-11-25 05:09 Sources read for the soft rule's unlearning (VISION.md): its authors ground it in earlier machine
  learning, not biology; the biological form (Bienenstock, Cooper & Munro 1982, reference verified) puts the
  sign on the cell's own activity against its own sliding threshold. **Owed after sh6: that form as a third arm,
  taken from the 1982 paper itself (open it first).** Artola et al. 1990 not opened: unverified, cited nowhere
  for readers.
- 2026-11-25 06:03 All healthy, 3.7 GB free. **The dissection's second step (seed 0):** own (each kind divides by its own
  largest change) destroys the layer (0.222; weights 90,000 long); early (the factor held at 1e-5) hardly
  develops (0.561 against 0.549 as born). The published layer needs a step shared by the layer that follows the
  changes' size: not local, and no stand-in of mine will be designed. **`organic-sh7`**
  (out/tmp/softhebb_dissect3_queue.sh; MemoryMax 1500M; about 3.5 hours): all five arms on seeds 1 and 2 →
  softhebb_dissect_s{1,2}.log, stream_{shh,shp,shc,she,sho}10k_s{1,2}.log. **Read tonight:** Blais et al. 1998
  (the quadratic BCM rule: its own stability, one cell). **The candidate for a fully local rule: BCM cells with
  lateral inhibition in the soft layer's pipeline; first read the network form (Intrator & Cooper 1992,
  Castellani et al. 1999) and take its constants from there.** softbase1: all three seeds' whole streams coded
  (equal to the kept codes); backprop of the soft shape without relay tuned (best 0.477 at 10,000: power 1,
  rate 0.001, batch 8); relay arms running; way back through the relay exact at power 1. Next: find and read
  the BCM network form; softbase1's relay tuning and seed 0's whole stream; sh7.
- 2026-11-25 06:07 **The BCM network form read in the paper** (Intrator & Cooper 1992; equations in VISION.md): the cell's own
  rule is local and stable by its sliding threshold; the network rule has a second term that carries other
  cells' modification signals (not obviously local: two arms, with and without); the paper gives no constants.
  **Next for queue E 27:** find the constants in a readable source (Law & Cooper 1994, PNAS 91:7797, PMC44489
  asks for a browser check; the authors' simulator; Cooper & Scofield 1988), then `rule="bcm"` in
  `softhebb_probe.py`'s pipeline, seed 0, stated expectation written first.
- 2026-11-25 07:18 All healthy, 3.1–3.7 GB free. **BCM's numbers found** in its authors' simulator (source read: eta 8e-6,
  tau 1000, births, sigmoid −1 / 50, 250,000 presentations, one cell, no lateral connections by default; the
  1997 NIPS paper opened, no numbers in it). **`experiments/bcm_probe.py`: a layer of 256 BCM cells that do
  not interact, in the soft layer's pipeline, seed 0.** Stable as stated. At the simulator's 250,000
  presentations it has hardly begun; at 2,500,000, 90 cells have developed and 79 of them are one
  colour-opponent blob or its opposite, two an oriented bar (2,034 near-copy pairs and rising;
  docs/img/bcm_filters_2500000_s0.png). **Running:** `organic-bcm4` (development to 10,000,000 →
  out/bench/bcm_develop_10000000_s0.log, MemoryMax 700M, about 30 minutes); `organic-bcm3`
  (out/tmp/bcm_queue.sh, MemoryMax 1500M, behind sh7 and bcm4: codes and streams of the layer at 250,000 /
  2,500,000 / 10,000,000 presentations and as born → bcm_code_s0.log, stream_{bcm,bcm_2500000,bcm_10000000,bcmr}10k_s0.log).
  **Dissection seed 1** agrees with seed 0 in structure (hard 930 near-copy pairs and 0.496; no unlearning
  19,033 pairs; const ran away; early 0 pairs; own 31,376 pairs, weights 72,000 long); its other streams and
  seed 2 run. **softbase1:** relay tuning at 19 of 24 (best with relay so far 0.528 at 10,000: power 0.7,
  rate 0.001, batch 32; without relay 0.477). Reward lives at 100,000 / 105,000: seeds 0 and 2 in preschool
  (P 0.01 / 0.00), seed 1 in pages 0.32; nothing new.
  **Next for queue E 27:** the stream readings (bcm3); then the reading for an inhibition whose strength is
  learned locally (Földiák 1990; the similarity-matching circuits) with BCM on the feed-forward synapses, and
  for the centre-surround input the BCM work used; nothing built before its source is read. softbase1's whole
  stream when it comes; sh7 seeds 1–2 into one sentence of the paper.
- 2026-11-25 07:20 **Read: Földiák 1990** (anti-Hebbian feedback whose strength is learned by a local rule; three rules and
  their numbers in VISION.md). BCM under that feedback would be my combination: not built. **Next for queue E
  27:** open Zylberberg, Murphy & DeWeese 2011 (a complete published layer with these local rules on natural
  image patches; open access), read rules and numbers, organic check, then that layer unchanged as the next arm;
  bcm4's reading at 10,000,000 presentations; bcm3's streams after sh7.
- 2026-11-25 08:34 All healthy, 3.7–3.8 GB free. **BCM at 10,000,000 presentations: ten different filters of 256** (as stated);
  the lengths go on growing and the activity is at the sigmoid's top, so "stable" is withdrawn as not shown
  (VISION.md). **Dissection seed 1 read:** same picture as seed 0 in every arm (hard 0.496, no unlearning 0.423,
  own 0.256, early 0.600 against published 0.684 and as born 0.546); seed 2 runs. **softbase1:** tuning done
  (best: no relay power 1 rate 0.001 batch 8, 0.477; fixed relay power 0.7 rate 0.001 batch 32, 0.528); **the
  soft layer on the whole stream, seed 0: word cells + relay 0.729 at 50,000 pictures, without relay 0.675**
  (our cortex in the paper's table: 0.693 ± 0.002); ridge, streaming LDA and backprop follow. **Read: Zylberberg,
  Murphy & DeWeese 2011 (preprint text) and the authors' code**: a complete published layer with Földiák's
  three local rules, spiking cells, every number given; organic check passed for an arm (p, the target rate,
  is the authors' set point). **`experiments/sailnet_probe.py`** written from the code; smoke test (1,000
  batches, another seed): on our raw colour windows the cells fire five times the target and pile onto one
  filter while still far from settled; the stated run (25,000 + 10,000 batches) is `organic-sail1`, its
  stream reading `organic-sail2`.
  **Next:** sail1's reading (if the layer does not settle on raw windows, the reason is the input's units:
  theirs is whitened and brought to unit deviation patch by patch; that input, in small, is then the second
  arm, with the response given the same input); bcm3 and sail2 streams; softbase1 seed 0 complete, then the
  paper's table; the lives' end at 120,000 (about 09:00) and the ramp lives' start.
- 2026-11-25 09:20 **ALERT read and mended in part:** the photo daemon wrote nothing for 31 hours because one of its two adults
  (`dots-studio/dots-3-note-preview:free`) was withdrawn by the provider (404 on every request since last
  night); removed from the text and curriculum daemons (restarted); the photo daemon waits for a second adult
  that passes the scan (WORLD.md; `organic-scan1`). The health alert stays until the photo world writes again.
  **SAILnet on the soft layer's raw windows: all 256 cells one filter** (my "fewer than 10 near-copy pairs" was
  wrong; the layer never reaches its targets; the rule's numbers are in the units of a whitened, mean-free
  patch). **Next arm, stated and running:** SAILnet in our cortex's place, on the cortex's own decorrelated
  windows (`experiments/sailnet_cortex_probe.py`, `organic-sailc1`/`sailc2`; smoke test at 1,000 batches: rates
  near target, but 1,265 near-copy pairs already). **Soft layer, whole stream, seed 0 complete for the local
  readers:** word cells + relay 0.729, ridge + relay 0.740, ridge 0.745, streaming LDA + relay 0.743, LDA
  0.741; **backprop of the layer's shape, no relay, filters learned from random: 0.547** at 50,000; the frozen
  and relay runs follow. Dissection seed 2 through its arms (same picture: hard 903 pairs, no unlearning
  19,610, const ran away, early 0 pairs, own weights 100,000 long); its streams run. Lives at 105,000–110,000:
  nothing new (seed 1 pages 0.34; seeds 0 and 2 preschool 0.02).
  Next: the scan's row (then the photo daemon's second adult, or the question to Kris); sailc1's reading; bcm3,
  sail2, sailc2 streams; softbase1's backprop rows for seed 0, then seeds 1–2 and the paper's table.
- 2026-11-25 10:03 Health: the photo alert stands until midnight (allowance spent); 3.2 GB free. **The photo world has its second
  adult again** (`inclusionai/ling-3.2-flash:free`, passed the scan; WORLD.md); check out/annotate.log after
  00:05 on 2026-11-26. **Dissection complete on three seeds** (table in VISION.md) **and in the paper** as a
  paragraph: the losers' unlearning does the work; the layer-wide division is not local and neither local
  stand-in works. **SAILnet in our cortex's place** after 25,000 batches: holds its target rates, but 92
  different filters of 256 (5,797 near-copy pairs; I had said fewer than 10); about one filter per input
  dimension (74). Final reading and streams still to come (sailc1, then sailc2 behind bcm3 and sail2).
  **softbase1, seed 0:** backprop of the soft shape 0.547 (no relay), 0.597 (fixed relay), frozen developed
  filters + gradient readout 0.623, against word cells + relay 0.729 and streaming LDA 0.743.
  Next: bcm3 / sail2 / sailc2 streams (about 11:40 for the last); softbase1 seeds 1–2 (about 16:00), then the
  paper's table and headline on the soft layer, draft 9 by email; the lives' end at 120,000.
- 2026-11-25 10:51 Photo alert stands until midnight; 3.5 GB free. **BCM in the stream, seed 0:** 0.443 (250,000 presentations) and
  0.371 (2,500,000) for word cells + relay, under the layer as born; the rest of bcm3 runs. **SAILnet in our
  cortex's place, final:** targets held, 98 different filters of 256; stream queued (sailc2, behind bcm3 and
  sail2, about 12:00). **softbase1 seed 0 complete:** frozen soft filters + gradient softmax readout + relay
  0.737, over word cells + relay 0.729 (PUBLISH.md says what that does to the headline; the frozen no-relay
  row used power 1, a caution in VISION.md); seed 1 whole stream: word cells + relay 0.734. Lives at
  110,000–115,000, nothing new. PUBLISH.md brought up to date.
  Next: the remaining streams; softbase1 seeds 1–2 → the paper's table, headline, draft 9; lives' end and the
  ramp lives' start (check memory then); photo log after midnight.
- 2026-11-25 11:42 Health: the photo alert (until midnight) and "kind_s1 stopped without a verdict": **kind_s1 ended as planned
  at 120,000 moments (11:10), still in pages** (W2 0.32, first words 0.44; no kill: 3.8 GB peak, a clean
  exit). The alert is the health check's wording for a life that ends by TOTAL inside a stage. **`organic-tag1`
  started its first arm, `tagn_s1`, at 11:38** when kind_s1 freed its memory (MemoryMax 3800M, dies before a
  life; 2.4 GB at start); the ramp lives start when kind_s0 and s2 end (about 12:30): **watch memory then**
  (three young lives + tagn + softbase1; tagn is the one the kernel takes first). **BCM and SAILnet-on-raw
  streams read** (table in VISION.md): every BCM state under its birth (0.443 / 0.371 / 0.471 against 0.566);
  SAILnet on raw windows 0.213. sailc2 is coding; its streams in about 30 minutes. **softbase1 seed 1:** word
  cells + relay 0.734, ridge + relay 0.751, streaming LDA + relay 0.753, backprop (random, no relay) 0.572.
  Next: sailc streams; tagn_s1's first checkpoint (the smoke test of continuing from a level); the kind lives
  read together into JUDGING.md when s0 and s2 end; softbase1 → paper.
- 2026-11-25 12:16 Health: photo alert (until midnight); kind_s1 and kind_s2 ended as planned at 120,000 (the "without a verdict"
  alerts are that), kind_s0 at 115,000; 6.0 GB free; `tagn_s1` runs (no checkpoint yet). **SAILnet in our
  cortex's place, stream: 0.484** (our cortex 0.613; I had said 0.63–0.67). Its filters: 107 cells one long
  filter, 75 alone, and **the short ones are oriented edges and gratings** (docs/img/sailc_filters_s0.png).
  **A fault found in the benchmark's input** (`experiments/upsample_probe.py`): CIFAR is shown blown up from 32
  to 64, and a 5 × 5 window of a blown-up picture holds 22 directions above the decorrelation's floor where a
  window of a real picture holds 47–48 (COCO at 64, CIFAR at 32). It bears on the paper's "developed filters do
  no better than random" and on the comparison of our cortex with the soft layer. **The control runs:**
  `organic-native1` (the same cortex and code on CIFAR at 32; expectation in VISION.md: as born 0.64–0.68,
  developed within 0.01 of it). softbase1 seed 1: backprop 0.572 / 0.611 (relay) / frozen no-relay 0.602.
  Next: native1's two streams; if as-born moves as stated, three seeds and the paper's benchmark has to be
  thought again (which input is the learner's); sailcu's stream; kind_s0's end and the three reward lives read
  together into JUDGING.md; the ramp lives' start (memory); softbase1.
- 2026-11-25 12:59 Health: photo alert (until midnight); the three "without a verdict" alerts are the reward lives ending as
  planned at 120,000 (12:20). **Reward lives read into JUDGING.md: a kinder reward changed nothing we can
  see**; preschool unlearned (0.04). Ramp lives started 12:20. **Native-size control, seed 0:** developed by our
  hard rule 0.656 against as born 0.627 (ridge + relay 0.667 / 0.633): development helps at native size, on
  one seed; I had said within 0.01. The blown-up as-born row on the same pictures was missing: added to the
  script (`SIZE=64`) and queued. **Memory:** with native2 running, available fell to 1.8 GB (my start of it
  left less than the rule allows); stopped native2 and native3 by name and replaced them with the gated
  `organic-native4`. 3.0 GB free now. `sailcu` 0.588. softbase1 seed 1 complete (frozen + relay 0.737), seed 2
  running.
  Next: native4's rows as memory allows → if three seeds agree, the paper's section on our development and
  the comparison with the soft layer are reworded (and which size is the learner's benchmark is a question
  for Kris); softbase1 seed 2 → table, headline, draft 9; tagn_s1's first checkpoint; ramp lives at 5,000.
- 2026-11-25 13:52 Health: photo alert (until midnight), the three planned endings; 2.8 GB free, so `native4` waits (gate 3.3 GB;
  it opens when softbase1 ends, about 17:30, or tagn_s1 does). **Soft layer, whole stream, local readers on
  three seeds** (table in VISION.md): word cells + relay 0.730 ± 0.004 (0.671 without), ridge 0.744, streaming
  LDA 0.748; backprop from random 0.562 ± 0.013 (three seeds), with relay 0.604 and frozen + relay 0.737 on two
  seeds so far. **Said beside it:** the soft layer developed on 50,000 other pictures first; the row that
  separates the two (backprop started from the developed filters) was not in the first queue: `organic-softbase2`
  queued. **tagn_s1 at 95,000:** continuing from a level works; two lives from one level differ by 0.05–0.07
  after 5,000 moments (first words 0.45 against 0.40, pages 0.25 against 0.18). Ramp lives at 5,000 (babbling).
  Next: softbase1 seed 2's last three rows (about 15:40) → the soft layer's rows into the paper's table as a
  second table with the pending row marked; native4 when memory opens; photo log after midnight.
- 2026-11-25 14:55 Health: photo alert (until midnight), the three planned endings; 2.7 GB free, native4 still waits. **softbase1's
  backprop rows complete on three seeds:** backprop of the soft shape 0.562 ± 0.013, with the fixed relay
  0.606 ± 0.008; frozen soft filters with a gradient readout 0.619 ± 0.016 (no relay, power 1) and 0.739 ± 0.003
  (relay), against the word cells' 0.730 ± 0.004. **Into the paper:** new section "The same contest on the better
  layer" (ten rows, the caveat about the 50,000 unlabelled pictures beside the number, one pending mark for the
  three rows still running); the old pending mark removed; abstract unchanged until softbase2. Builds, 19
  pages. tagn_s1 at 100,000: 0.42 / 0.23 / 0.00 against plat_s1's 0.46 / 0.23 / 0.01 (the two lives cross:
  test noise). King et al. 2013 could not be fetched (journal blocks, no preprint): not opened.
  Next: shrfull's three seeds (about 17:45) → as-born row, **email draft 9**; native4 when memory opens
  (softbase1's end frees about 0.9 GB); softbase2 behind both; photo log after midnight.
- 2026-11-25 15:55 Health unchanged (photo alert until midnight; the three planned endings); 2.6 GB free, native4 waits. **The soft
  layer as born, whole stream, seed 0** (stream_shrfull_s0.log): word cells + relay 0.633, without relay 0.493;
  ridge 0.658 / 0.640; streaming LDA 0.663 / 0.655. One seed: with the relay the word cells on never-developed
  filters are above backprop with the relay (0.597 on that seed), and without it they are below backprop
  (0.547): the order our cortex showed. Seeds 1–2 about 17:40. Ramp lives at 15,000, in first words (0.27,
  0.25, 0.35). **Paper, a methods omission closed:** the Setup now says the eye is 64 pixels and CIFAR is enlarged
  to it (and that the soft layer sees the pictures at their own size); 20 pages.
  Next: shrfull seeds 1–2 → the as-born rows into the second table, email draft 9; native4; softbase2.
- 2026-11-25 16:57 Health unchanged (photo alert until midnight; the three planned endings); **2.4 GB free** (ramp lives growing;
  nothing started). **The soft layer as born, whole stream, three seeds for the word cells:** 0.630 ± 0.004
  with the relay, 0.503 ± 0.010 without (backprop 0.606 / 0.562): two points ahead with the relay, six behind
  without; the soft rule's development is worth 0.100 / 0.168 to the word cells. Into the paper (two rows, one
  paragraph; pending mark now only softbase2's rows). **Draft 9 emailed to Kris 16:55.** tagn_s1 at 105,000:
  0.40 / 0.19 / 0.02 (plat_s1: 0.42 / 0.25 / 0.00). Ramp lives at 20,000, first words 0.33 / 0.31 / 0.31.
  Next: shrfull seed 2's second-order rows (17:30) into VISION; softbase1 and tagn_s1 end → native4 opens;
  watch memory (if available falls under 2.1 GB with nothing of mine to stop, softbase2 and native4 stay
  gated; the tag queue's next arm needs 5.7 GB and will wait); photo log after midnight.
- 2026-11-25 17:45 Health unchanged (photo alert until midnight; the three planned endings). `organic-softbase1` ended 17:26 and
  `tagn_s1` at 110,000: **5.1 GB free**, native4 is counting its three minutes at the gate; softbase2 behind it;
  the tag queue's next arm needs 5.7 GB and will wait behind both (about ten hours: the paper's rows first).
  **As born, second-order readers, three seeds:** ridge + relay 0.666 ± 0.008, streaming LDA + relay 0.663 ± 0.003
  (one sentence into the paper). **tagn_s1 whole:** over four tests its means are 0.405 / 0.218 / 0.015 against
  plat_s1's 0.413 / 0.213 / 0.013; single tests differ by up to 0.07 → the replay arms are read on the mean of
  four tests; under about 0.03 says nothing. Ramp lives at 25,000 (seed 1 at 20,000): first words 0.36 / 0.31 / 0.42.
  Next: native4's seeds 1–2 (about 45 minutes a step) → three-seed native table → the paper's section on our
  development and the question for Kris; then softbase2; photo log after midnight.
- 2026-11-25 18:55 Health unchanged (photo alert until midnight; the three planned endings); 4.5 GB free. **Native-size control,
  three seeds: our hard rule's development helps** (word cells + relay 0.649 ± 0.008 against 0.626 ± 0.002;
  ridge + relay 0.666 ± 0.001 against 0.632 ± 0.003; all twelve differences positive; I had said within 0.01).
  native4 is on the enlarged like-for-like rows (`big_r`, `big_d`, three seeds, about 21:15). **Started
  `organic-sailn1`:** SAILnet in the cortex's place at native size (switch `SIZE=32` in
  sailnet_cortex_probe.py, default unchanged), expectation written first. Ramp lives at 30,000: first words
  0.39 / 0.39 / 0.43.
  Next: sailn's development log (about 19:50) and streams; big rows → the paper's section on our development
  reworded with both sizes; softbase2 after native4; photo log after midnight.
- 2026-11-25 19:55 Health unchanged (photo alert until midnight; the three planned endings); 4.1 GB free. **SAILnet at native
  size, seed 0: the collapse is gone** (197 different filters of 256, rates on target; 98 on enlarged pictures)
  and its filters read 0.641 / 0.661 (word cells + relay / ridge + relay, unit length) beside as born 0.627 /
  0.633 and our hard rule 0.656 / 0.667. What I said before held, for once. Seeds 1–2 started
  (`organic-sailn2`). New `experiments/filters_picture.py`; three pictures in docs/img. Enlarged rows, seed 0:
  development −0.014 (word cells) / +0.012 (ridge): the paper's "did nothing" holds on enlarged pictures.
  Next: sailn seeds 1–2 (about 21:30) → three-seed row; big rows seeds 1–2 (about 21:15) → the paper's section
  with both sizes; if SAILnet holds on three seeds: its own answer (spikes under inhibition) as the layer's
  response, and whether it belongs in the paper beside the hard rule; softbase2; photo log after midnight.
- 2026-11-25 20:40 **Kris (20:20): use all kinds of pictures, not only CIFAR** → standing rule 10; memory and memoize written.
  On disk already, real at the eye's size: 95,249 labelled COCO crops (35 words), 3,482 Wikimedia Commons
  pictures (16 words), 118,287 unlabelled COCO photographs. **Built the photo stream:**
  `experiments/bench_photos.py` (33 words with at least 1,000 crops; a twentieth of the photographs held out
  whole: 4,884 tests, 170 of them from Commons; stream 50,000, 1,747 from Commons; the cortex develops on the
  115,475 unlabelled photographs that are not test photographs); `bench_stream.py` gains `SET` (default
  cifar10, unchanged; edited atomically while queues use it). The word cells say all 33 words (checked).
  **Started `organic-photos1`** (seed 0). Also read on the way: SAILnet native seed 1 has 201 different
  filters, word cells + relay 0.625 as developed (seed 0: 0.633); enlarged rows seed 1: as born 0.651 / 0.661,
  developed 0.619 / 0.662. Health unchanged; 3.5 GB free with native4, sailn2 and photos1 gated.
  Next: photos1's code logs (about 21:30) and streams; sailn2 and native4's last seeds → three-seed tables;
  the paper's section on our development with both sizes, and a place for the photo stream; softbase2.
- 2026-11-25 20:55 Health unchanged; 3.9 GB free. photos1 is coding the stream (ph_r). **SAILnet native, seed 1:** 201 different
  filters; unit length 0.636 / 0.652 (word cells + relay / ridge + relay) beside as born 0.627 / 0.628 and the
  hard rule 0.640 / 0.666: above born on both seeds for ridge (+0.028, +0.024), under the hard rule by 0.006
  and 0.014. **The photo stream's reference queued** (`organic-photos-bp1`): `SET=photos` added to
  bench_backprop.py (pictures, labels, order and unlabelled pictures of the photo stream; default unchanged).
  Next: photos1's logs; the tuning log's first line (traceback?); sailn seed 2, big seed 2 → three-seed tables
  and the paper's section; softbase2 after native4.
- 2026-11-25 21:35 Health unchanged; 3.5 GB free. `organic-native4` ended 21:00; `organic-softbase2` started 21:04. **Both sizes,
  three seeds, same 10,000 pictures:** enlarged, development −0.022 for the word cells (every seed) and +0.009
  for ridge; own size, +0.023 and +0.034. **Into the paper** (two-size table and paragraph; the soft-layer
  comparison corrected to 0.689 against 0.649; abstract qualified; 21 pages). Open: 658 near-copy pairs in the
  paper against 257–282 here. SAILnet native: 197 / 201 / 192 different filters; seed 2's unit-length stream
  running. Photo stream: ph_r coded, ph_d in work, then the streams and the backprop tuning.
  **Beside the loop (private, not in this repository):** Kris asked about ANID's Startup CTCI 2027 and a pilot
  of the learner in his startup; notes are in ~/franky-anid/NOTES.md, to be taken up when he says.
  Next: photo streams (first numbers about 23:00) and backprop_photos_tune_s0.log's first line; sailn seed 2 →
  three-seed row in VISION and, if it holds, a sentence in the paper; softbase2's rows (about 03:00); photo
  daemon's log after midnight; draft 10 when the photo stream has its table.
- 2026-11-25 22:30 **Kris: "yes slow polite download constantly"** → `organic-commons` installed and running (WORLD.md). Photo
  stream, seed 0, as born: word cells + relay 0.439 after 50,000 (0.391 at 10,000), without relay 0.379;
  ph_d coded (177 different filters, 180 near-copy pairs); backprop's first tuning run on photos 0.301 at
  10,000 (the code path works). Kris also asked whether the enlargement needed reporting: in the paper it
  shrinks to a sentence and a table once the photo stream is the vision evidence.
  Next: as before (photo streams and tuning; sailn seed 2; softbase2; photo daemon after midnight).
- 2026-11-25 22:40 Health unchanged; 3.8 GB free. **SAILnet native, three seeds:** 0.643 ± 0.008 / 0.657 ± 0.005 at unit length
  (as born 0.626 / 0.632; hard rule 0.649 / 0.666): a fully local rule at the hard rule's level → a paragraph
  in the paper (21 pages built). `organic-sailn2` ended. **softbase2 seed 0:** backprop started from the soft
  filters 0.701 with the relay (frozen 0.737, word cells 0.729, from random 0.597) and 0.606 without. Photo
  stream as born: 0.439 after 50,000. `organic-commons` at 74 pictures.
  Next: photo streams ph_r's second-order rows and ph_d; the tuning; softbase2 seeds 1–2 (about 01:30) → the
  pending row of the paper's second table and the abstract; the photo daemon's log after midnight.
- 2026-11-25 23:35 Health unchanged; 3.8 GB free. **Photo stream seed 0:** as born 0.439 (word cells + relay), 0.459 ridge, 0.476
  streaming LDA; **developed 0.443: development gives the word cells nothing on real photographs** (I had said
  +0.01–0.03); the crops are not enlarged (checked), so the CIFAR-at-32 gain does not generalise as I framed
  it; ridge and seeds to come. Tuning of backprop on photos: 0.30 at 10,000 so far. softbase2 seed 1: 0.708 /
  0.615 / 0.583. Ramp lives at 50,000, in first words. Commons: 1,432 pictures.
  Next: photo daemon's log after 00:05; ph_d's ridge rows; tuning; softbase2 seed 2 (about 00:15) → the paper's
  pending row on three seeds and the abstract; then photos seeds 1–2 (queue after photos1).
- 2026-11-26 00:45 Health: only the three planned endings now; **the photo daemon writes again** (15 of 20 judged right; two
  batches without a reply). 4.1 GB free. **Photo stream seed 0 whole:** development +0.023 for ridge and
  streaming LDA behind the relay, +0.004 for the word cells: the second-order readers gain, ours does not, as
  on enlarged CIFAR. photos1 ended; **photos2 (seeds 1–2) and sailp1 (SAILnet on photographs) started.**
  softbase2 seed 2: 0.701 with the relay (three seeds 0.701, 0.708, 0.701); two rows to go. Tuning on photos:
  batch 32 worse than 8 so far (0.267, 0.165). Commons: 2,690.
  Next: softbase2's end → the paper's pending row, the abstract; photos2 and sailp1 as they come; the tuning's
  fixed-relay half; the ramp lives into JUDGING.md at 60,000.
- 2026-11-26 01:40 Health: the three planned endings only; 3.3 GB free (photos2, sailp1, photos-bp1 gated). **softbase2 ended,
  three seeds:** backprop handed the soft filters 0.703 ± 0.004 with the relay (random start 0.606; frozen
  0.739; word cells 0.730), 0.620 ± 0.017 without. **The paper's second table is complete, its reading and the
  abstract rewritten; no pending mark left in that section** (21 pages). SAILnet on photographs developed 145
  different filters (I had said at least 180). Photo tuning: 7 of 12 runs; with the relay backprop reaches
  0.403 at 10,000 (word cells 0.391). Photo daemon: 69 requests, last batch 20 of 20 judged right. Commons: 4,014.
  Next: photos2 (seed 1's streams about 03:30), sailp1's codes and streams, the tuning's end and the two whole
  streams → the photo stream's table (three seeds of ours, one of backprop) → a section in the paper → draft 10
  to Kris in the morning; ramp lives into JUDGING.md; the tag queue's arm opens when memory does.
- 2026-11-26 07:40–08:10 (the loop was idle 02:28–07:37: the model allowance ran out; Kris switched the model). Health: the three
  planned endings; 3.4 GB free. **Photo stream, three seeds:** development +0.025 for ridge behind the relay on
  every seed, −0.001 for the word cells (−0.016 without the relay). Backprop on photographs, seed 0, no relay:
  0.387 (its best setting at the search's edge → `organic-photos-bp2` widens it). **SAILnet on photographs:**
  145 filters; 0.427 / 0.474 / 0.498 (word cells / ridge / streaming LDA, relay). **Hypothesis from five codes:**
  the word cells gain from development only where the developed code spreads over about 500 directions after the
  relay → `organic-relaymore1` (relay settled on 20,000 photographs). **Photo daemon:** 27 of 33 batches failed
  since midnight, every retry counted → its share gone at 04:02; failures now name the model and the reason
  (organic/adults.py, logging only; photo daemon restarted, asleep till midnight). Ramp lives at 80,000: no
  visible effect. Commons: 11,807. **Kris asked about video** → queue E 31 (above).
  Next: photos-bp1's relay row → the paper's photo section → draft 10; relaymore1; bp2; photo log after midnight;
  E 31 (a): licences and sizes of real video.
- 2026-11-26 08:20 **Kris: include video; the future use is streaming data; download Wikimedia video in parallel; 1 TB of disk,
  the machine is ours alone.** CORe50 fetching (about nine hours at the server's pace); `organic-commons-video`
  installed and running (WORLD.md). Memory and memoize written.
- 2026-11-26 09:15 Health: the three planned endings; 3.6 GB free. **Backprop on photographs with the relay: 0.462** (seed 0;
  word cells 0.446 ± 0.006; ahead early, behind by two at the end). Photo stream complete on three seeds. **The paper
  has a section "Real photographs at the eye's size"** (nine rows; backprop's seeds 1–2 pending). **Relay on 20,000
  pictures: nothing moves** (396 directions, word cells 0.441): sample size ruled out. **Spectrum probe:**
  development halves the code's spread after the relay everywhere; the word cells gain only where it stays near 900
  → the relay's floor (relative to the largest variance) queued as the next diagnosis. Artifact for Kris: the two
  architectures step by step (https://claude.ai/artifact/NLZYiDC19SLy3UHpsvRHwL). Commons video: 234 videos,
  106,510 frames; CORe50: 0.97 of 5.9 GB.
  Next: relaymore1's ph_r arm and relayshrink1; bp2's last three runs; backprop on photographs, seeds 1–2 (queue
  when bp2 ends); photo daemon's log after midnight; ramp lives at 120,000 (evening).
- 2026-11-26 09:55 Health: the three planned endings; 3.8 GB free. Relay on 20,000 pictures: nothing moves for the as-born
  code either. Spectrum at a five-times-lower floor predicts the developed photo code spreads like CIFAR-at-32's
  (1,000 directions) → relayshrink1 (running after relaymore1) is the test. **New organ: a cochlea** (64 ERB-spaced
  bands, 10 ms frames, power 0.3; tones checked) and the service that hears the Commons videos through it. Backprop's
  widened tuning: no wider setting wins (one run left). `experiments/growth_census.py` (numbers each organ holds,
  stage by stage) running on plat_s0's levels. Commons video: 758 videos, 409,157 frames; CORe50 2.2 of 5.9 GB.
  Next: relayshrink1; the census into GROWTH.md; bp2's end → backprop on photographs, seeds 1–2; then E 32's first
  probe (audio and video teaching each other by a local rule) once a few thousand videos with sound are in.
- 2026-11-26 11:45 Health: all healthy (the reward lives' planned endings are now in out/monitor/ended.json). **The relay's
  floor is not the reason** the word cells take nothing from development on photographs (VISION.md 11:40): a five
  times lower floor lifts both codes by a point and leaves development's gain at +0.004 (ridge +0.021); my
  prediction was wrong. Next suspect: the reader → `organic-ncm1` (nearest class mean, three seeds). Backprop's wider
  search: nothing wins; seeds 1–2 queued (`organic-photos-bp3`). **E 32's first probe running** (`organic-av1`,
  experiments/av_probe.py: does a frame predict its own sound on held-out videos; prediction in VISION.md 11:10).
  Census with word-cell turnover queued for six lives. CORe50 frames converter written (zip half fetched). Paper:
  the 658-pairs sentence names its threshold.
  Next: av_probe seed 0 → read against the prediction; bp3; ncm1; census; CORe50 frames when the zip is whole.
- 2026-11-26 12:25 Health: all healthy. **av_probe seed 0 killed at its own cap (1.7 GB of 1.8) at 11:53** in the readers;
  made lean (blocks in place, no copy of the training set, the cortex freed, codes cached in out/bench/av_codes_*.npz,
  the video set frozen in out/bench/av_pages.json for all seeds; smoke test reproduces the numbers from the cache) and
  re-queued behind a 3.3 GB gate. **Growth: word-cell turnover on six lives** (queue E 33 above; docs/GROWTH.md): half
  the cells replaced between babbling and echo, then turnover falls tenfold and a stable old core forms. ncm1 running
  (census was quick); bp3's first run running.
  Next: ncm1 → the reader's verdict; av1 when there is room; bp3; CORe50 frames when the zip is whole.
- 2026-11-26 14:00 Health: all healthy; 2.6 GB free (three probes and the ramp lives). **Reader verdict, seed 0:** the
  nearest class mean loses two points from development on photographs (0.419 → 0.401) while ridge and LDA gain two:
  development's worth is in the code's second moments (VISION.md 13:20); seed 1's as-born 0.422. **Artifact v2** for
  Kris's talk (per-part details, stand-ins, 77 checked references). **Kris: sound without transcription; an
  asynchronous brain** → SpokenCOCO fetching (CC BY-SA), queue E 32 and E 34 written; serial-gaze continuation lives
  queued (`organic-gaze-lives`). **CORe50 frames done** (164,866, checked by eye) → `organic-core50-1` queued behind
  ncm1 with a smoke test first. Shared memory gate for new queues (out/tmp/gate.sh).
  Next: av_probe seed 0's readers; ncm seeds; bp3; CORe50 probe; ramp lives' end tonight → gaze and tag lives.
- 2026-11-26 15:50 Health: all healthy. **Kris: neurons, not turns** → docs/ASYNC.md rewritten around a stream brain on a
  10 ms clock; **`organic/stream.py` runs** (eye race 3.3 saccades/s, fixations median 290 ms, own rhythm; mouth
  babbling concurrently; 6.5 CPU-s per simulated second). **Two senses, seed 0:** Hebb 0.61–0.63 picks the true sound
  (chance 0.5), above delta and least squares; developed hearing +1 point over born (VISION.md 15:45). **Sound ear:**
  first design failed its smoke test; three levels (rhythm syllables, eye-rule kinds, byte-ear words) running on the
  val part (10,000 of 20,028 heard at 15:45). docs/ASYNC.md design. CORe50 probe queued behind ncm1.
  Next: av seed 0's word test, then seeds 1–2; sound units' first reading; ncm seeds; bp3; ramp lives' end (~19:00)
  → gaze and tag lives; stream brain: cheaper link, sound pulling the eye, babbling heard.
- 2026-11-26 17:10 Health: all healthy. **Reader verdict, three seeds**: nearest class mean loses two points from
  development on photos (0.421 → 0.403) → paper's photo section updated (second-moment reading; relay explanations
  ruled out). **Two senses, seed 0, word test**: sight 0.239, hearing 0.201, both 0.264 (VISION.md); seeds 1–2 running
  (`organic-av2`). **Sound ear, first reading**: syllable scale right, nothing recurs → second reading running
  (adaptation, 8 bands, window kinds, lift). **Stream brain**: a vocal tract; babbling learns an auditory-motor map
  (29/32 own voice; neighbours in another); heard vowels draw near imitations 0.22 vs 0.06 (VISION.md 17:05).
  Next: sunits2; av seeds; CORe50 probe; ramp lives' end (~19:00) → JUDGING.md, gaze/tag lives start; stream brain:
  echoic trace for imitation, habits from what is heard.
- 2026-11-26 18:25 Health: all healthy; memory at the floor (2.2 GB) until the ramp lives end. **Stream brain imitation,
  three seeds:** real and modest (rank 13.1 against 16.3; nearest three 0.18 against 0.09); the echoic trace adds
  nothing (two forms tried; off). `experiments/reward_table.py` reproduces the kind lives' table (ready for the ramp
  lives). Backprop on photographs, seed 1: 0.464 with the relay, 0.390 without. The sound ear's second reading runs
  about seven times slower than the first on this crowded machine (ends near midnight).
  Next: the ramp lives at 120,000 → JUDGING.md (reward_table.py) and the census; then the gaze and tag lives start.
- 2026-11-26 19:55 **Ramp lives ended and read** (JUDGING.md: three reward schedules, nine lives, no difference; a
  hint that pages erode more under the ramp). Ended lives marked (out/monitor/ended.json). **gazs_s1 started 19:25**
  (the first serial-gaze life, continued from plat_s1_W2; running normally, no records yet); the tag queue now waits
  behind the gaze lives (`organic-tag2`). Two senses seed 1 agrees with seed 0 (together > sight). Census of the
  ramp lives queued (`organic-census3`). CORe50 seed 0 running (first stage: same object elsewhere 0.117).
  Next: gazs_s1's first record (about 21:00); av seed 2; CORe50 arms; sunits2 (about midnight); photo daemon's log
  after midnight.
- 2026-11-26 20:50 Health: all healthy. **Stream brain: babbling drifts toward the language heard** (exploratory, three
  seeds, 6 of 6; VISION.md 20:45), after gating habits off one's own voice. CORe50 seed 0: first stage alone same
  object elsewhere 0.117, the second stage in order with the trace 0.116 (shuffled and plain running). Ramp census
  in (same turnover pattern). gazs_s1: no record yet at 85 minutes (first due at 95,000).
- 2026-11-26 21:25 Health: all healthy. **gazs_s1's first record (95,000, 5,000 moments of serial gaze after a life of
  looking all at once)** against its control tagn_s1 at the same age: naming on pages 0.263 against 0.285 (6.6 ticks
  against 5.2), "is there" on pages 0.250 against 0.243 (3.5 against 2.4 ticks), first words 0.439 against 0.454,
  pages 0.231 against 0.247: level within a few points, a tick slower, as on the small learners. One seed, early.
  Backprop on photographs with the relay, three seeds: 0.462 / 0.464 / 0.462 (no relay: 0.387 / 0.390 / seed 2 running).
  Owed for Kris's talk: the artifact gets a short "being built now" part (two senses, the stream brain, babbling)
  once tonight's seeds are in.
- 2026-11-26 21:55 **Two senses, three seeds: together-beats-sight does not hold** (seed 2: 0.218 against sight 0.254;
  corrected in VISION.md, memoize, PUBLISH.md; told Kris at 19:55 too early). What holds: Hebb 0.60–0.64 on the true
  sound; any hearing cortex above the cochlea; its development about a point. **Backprop on photographs, three seeds**
  (0.463 ± 0.001 with the relay, 0.392 ± 0.005 without) → paper; **draft 10 built and emailed** (five mistaken
  assumptions, the development section retitled). CORe50 seed 0: order 0.116, shuffled 0.119, first stage 0.117 (plain
  running): time makes no difference to this second stage so far.
- 2026-11-26 22:45 Health: all healthy. **CORe50 seed 0 read** (VISION.md 22:25): order, shuffled and no trace level in the
  3 × 3 second stage (0.113–0.119; first stage 0.117); my prediction was wrong. A whole-view (IT-like) trace stage built
  and run on three seeds: too weak to test (0.03–0.04 against its input's 0.117): inconclusive, left in the queue.
  gazs_s1 at 100,000: pages 0.176 against its control's 0.232, first words level (one reading; both lose earlier
  stages in preschool). Sound ear second reading: all 20,000 heard, held-out reading running.
- 2026-11-26 23:10 **The text-free ear finds word-like units** (VISION.md 23:10: 111 kinds against 0 shuffled; "group",
  "woman", "people", "computer"). SpokenCOCO download done (67.8 GB); its training part through the cochlea running
  (`organic-scoco-train`, about an hour). **gazs_s1 at 105,000: first words fell to 0.244** (control 0.397), on every
  source, text-only questions too (0.24 → 0.06), answers slower: the shared gate unsettled by the new eye commands?
  One life; watch 110,000 and the other seeds; if it holds, the serial gaze belongs from birth, not switched on
  mid-life. CORe50 streams of seed 0 done (to read).
- 2026-11-27 00:00 SpokenCOCO's training part: the cochlea pass was killed at its cap at 559k of 617k (a streaming tar
  keeps every member it passed); fixed, index and store reconciled, resumed. `experiments/grounded_speech_probe.py`
  written (the ear's own sound kinds against the photographs by Hebb's rule; the paradigm's retrieval test, R@1/5/10
  among 1,000 held-out pairs, with a shuffled control; no transcripts read at all); smoke test when the pass ends.
  gazs_s1 at 110,000: first words 0.297 (control 0.352), pages 0.184 (0.201): partly back. CORe50 seed 1: first stage
  0.120, in order 0.109 (shuffled and plain running).
- 2026-11-27 00:40 Health: all healthy. **SpokenCOCO through the cochlea, complete** (617,222 recordings, 267 M rows).
  **Photo daemon, read after midnight:** both failures tonight are inclusionai/ling-3.2-flash:free answering HTTP 429
  (the free provider's rate limit; ling does take pictures: the scan of 2026-11-25); the daemon used to retry into
  the limit and drop the batch: now it cools down 30 minutes and tries the same photos again (annotate.py; restarted).
  **`organic-grounded1`** queued (smoke test, then seed 0: the ear's sound kinds against the photographs, R@1/5/10).
  CORe50 seed 1: first 0.120, order 0.109, shuffled 0.102 (plain running).
- 2026-11-27 01:20 Health: all healthy. Grounding smoke test passed (chance at 600 captions, as expected); the full run
  (`organic-grounded1`, seed 0, 20,000 + 20,000 captions, 1,000 test pairs) is running: some ten hours. CORe50 seed 1
  read (VISION.md): the order does not help the 3 × 3 stage on two seeds. Photo daemon: ling still answers 429; the
  cool-down retries every 30 minutes (two requests a try, no photos dropped).
- 2026-11-27 01:50 **Artifact v3 for Kris's talk** (https://claude.ai/artifact/NLZYiDC19SLy3UHpsvRHwL): a "Being built now"
  section (a 1.5 s timeline of the stream brain; five result cards with seeds and status: babbling imitation (3 seeds),
  babbling drift (3 seeds, exploratory), the text-free ear's word-like units (1 seed), sight and sound (3 seeds, the
  combination claim marked not established), time in video (2 seeds, negative); Crossref-checked references), and
  the photo row of the chart on three seeds of backprop. Li & DiCarlo 2008 checked on Crossref.

- 2026-11-27 03:25 Health: all healthy. **Commons video daemon was killed at its 800 MB cap every ~12 minutes since
  00:32** (nine times): it read each video whole into memory and one was too long, fetched again at every restart. Now
  streamed to disk a megabyte at a time, files over 300 MB skipped (the largest kept so far 244 MB); restarted 02:22,
  past the file (peak 640 MB). **gazs_s2 ended at 120,000:** first words 0.449 and pages 0.199 against plat_s2's own
  0.481 and 0.186 at that age, never the collapse of gazs_s1 at 105,000 (control tagn_s2 still to come); gazs_s0
  started 02:39. **CORe50, two probes (VISION.md 02:25, 02:55):** the slow stage (Hebb on a trace, subspace rule)
  learns nothing from order on three seeds (prediction failed); the reason measured: the whole-frame code is ruled
  by the scene (another kind in the same session 0.15 alike, the same object elsewhere 0.03). **The eye on the thing,
  seed 0: the middle fixation lifts the same object elsewhere from 0.117 to 0.164** (the scene's share down by two
  thirds); the reflex's salient pick does worse (0.075). Queued seeds 1–2 and streams (`organic-fovea-q`); the eye
  drawn by motion on seed 0 running. Photo daemon: ling still 429 since midnight (cool-down working; no photos lost).
  Grounding run: the ear has heard 15,000 of 20,000 captions (about three more hours).
- 2026-11-27 03:40 **The eye drawn by motion, seed 0: 0.093** (between the salient pick's 0.075 and the whole frame's
  0.117; it jumps to the hand); the middle fixation seed 1: **0.164** against the whole frame's 0.120 (two seeds now,
  +0.047 and +0.044). Reading (VISION.md 02:55): an eye on the held thing helps a lot, but neither reflex finds the
  thing; the middle is where CORe50's crop puts the object. Next for E 34: pursuit (a fixation that persists and
  follows), measured first by how often it lands on the middle. Timestamps written ahead of the clock tonight
  (03:30, 03:55, 04:00) corrected to the real times.
- 2026-11-27 03:45 Seed 1 of the fixation: the middle 0.164 against the whole frame's 0.120 (two seeds); the whole frame
  beside it adds nothing (0.162). **Pursuit measured** (`experiments/core50_pursuit_probe.py`, where the eye lands, no
  code read): within 8 px of the object 0.26 jumping, 0.34 pursuing, **0.45 pursuing motion against the view's own flow**
  (the hand-held camera moves the whole background). Reading: reflexes get the eye on the thing half the time at best;
  the rest is for a learned value of places (E 23). Line closed for tonight; the queue's streams on the fixated codes
  will say what the middle fixation is worth to the word cells.
- 2026-11-27 04:53 Health: all healthy. **The fixation holds on three seeds:** the middle 0.164 / 0.164 / 0.168 against
  the whole frame's 0.117 / 0.120 / 0.118 (memoize). **But the word cells do not take it** (VISION.md 04:48): on the
  stream of 10 kinds the ridge gains +0.025 / +0.028 on the middle fixation and the word cells lose 0.015 on both
  seeds so far; my first explanation (neighbouring frames) was wrong, the stream is shuffled; nearest class mean and
  streaming LDA queued to tell the two remaining readings apart (`organic-ncm-c50`). CORe50 seed 2 of the 3 × 3
  stage: first 0.118, order 0.113, shuffled 0.099 (plain running): the order does not help on three seeds. gazs_s0 at
  115,000: first words 0.52, pages 0.17 against plat_s0's 0.53 and 0.13 at that age (no collapse). Grounding: the ear
  has heard all 20,000 captions; the bags next. Commons video: no restart since the fix (its 800 MB is mostly page
  cache, 69 MB its own). **Photo daemon: nothing added for 25 hours** (ling 429 all night); scan of two more free
  models that look; particle-1 taken as a spare that stands in for a rate-limited adult (WORLD.md 04:52).
- 2026-11-27 05:02 **Photo daemon flowing again:** with particle-1 standing in while ling rests, the first two batches
  were judged 19 / 20 and 20 / 20 right. **Two reflexes for the stream brain's eye** (E 34; Kris's "do they move at
  all?"): what moves against the view's own flow draws saccades (`motion_drive`), and between saccades the eye
  follows it (`pursuit_tau`, 100 ms); both off by default (digest of 3 s of the default eye identical before and
  after). Said before it runs (VISION.md 04:57); `organic-stream-eye1` measures where the eye lands and what it then
  recognises across sessions, three seeds. Tonight's four references (Yu & Smith 2012, Lisberger 2010, Wiskott &
  Sejnowski 2002, Sprekeler et al. 2007) found on Crossref with their DOIs (content still from memory).
- 2026-11-27 06:04 Health: all healthy (load 26 on 16 cores: the stream-eye probe runs at a tenth of its smoke-test
  speed; seed 0's first arm took 35 minutes). **CORe50 on three seeds** (VISION.md 06:03; memoize): time order
  teaches no recognition across scenes with either second stage; on the middle fixation order beats shuffled
  (+0.007 every seed) but not plain Hebb (a trace over shuffled frames hurts). The word cells lose on the middle
  fixation on three seeds (−0.015, −0.015, −0.005) where ridge gains (+0.025 to +0.028) and, on seed 0, the means too
  (nearest class mean +0.022, streaming LDA +0.023): it is the cells, not first-order reading. Queued: the cells as in
  the lives (corrections scaled by their own error) on both codes (`organic-cells-err`). The stream brain's eye, seed
  0, today's eye: 0.071 (21% within 8 px of the middle). gazs_s0 at 120,000: first words 0.47, two words 0.47, pages
  0.17 against plat_s0's 0.49, 0.46, 0.15 (level).
- 2026-11-27 06:05 **Artifact v4 for Kris's talk** (https://claude.ai/artifact/NLZYiDC19SLy3UHpsvRHwL): a new card "An eye on
  the held thing" (3 seeds: 0.165 against the whole view's 0.118, with the crop caveat, the reflexes' misses and the
  word cells' miss), "Time in video" now on 3 seeds with both second stages, and three references added from Crossref
  (Yu & Smith 2012, Lisberger 2010, Wiskott & Sejnowski 2002; one-line summaries from memory, for Kris to open).
- 2026-11-27 07:11 Health: all healthy (load 11; 7.0 GB free). **The cells' corrections scaled by their own error**
  (as in the lives; VISION.md 07:11; memoize): they lose as much on the middle fixation as full-size corrections
  (−0.016, −0.012 against −0.015, −0.015) while their memory of a frame seen once falls by 0.13–0.17: recency is not
  why the cells miss the gain. Seed 1 of the means agrees with seed 0 (both take the gain). Queued a diagnostic delta
  reader, the cells' rule without naming by units (`organic-delta-c50`; smoke test: cells 0.190 as before, delta
  0.192 at 500 frames). **The stream brain's eye, seed 0** (memoize): pursuit lifts what is recognised elsewhere from
  0.071 to 0.123, both reflexes to 0.140 (above the whole frame's 0.117; the middle 0.168); motion-drawn saccades alone
  0.082. What I said at 04:57 held but the within-8-px share (0.34 against 0.35). The probe was killed for memory in
  seed 1 (a reference cycle per video; fixed) and restarted as `organic-stream-eye2`. Seed 2 of the time-variant
  streams agrees (order 0.373, shuffled 0.370, no trace 0.374). gazs_s0 still at 120,000 (125,000 due).
- 2026-11-27 08:30 Health: all healthy (load 3; 8.4 GB free). **Why the word cells miss the fixation's gain, by
  elimination on three seeds** (VISION.md 08:28; memoize): a delta reader with the cells' rule and no words loses
  as they do (−0.012 on every seed), so not naming by units; ridge on the cells' visual neurons' pattern is ridge
  on the relayed code to the third decimal, so not the second normalisation; **the relay settled on COCO** is: on
  CORe50 the relayed code is far from even (participation ratio 131–137 whole, 282–291 middle, of 4,096), and
  evened again on the stream's own frames (no labels) the delta reader takes the gain in full (0.398 / 0.400 / 0.409
  middle against 0.356 / 0.364 / 0.360 whole: ridge's level). Queued the brain's way of doing that, the relay
  circuit learning along the stream (`organic-relaylearn`; said before it runs). **The stream brain's eye on three
  seeds** (VISION.md 08:29): pursuit 0.123 / 0.126 / 0.125, both reflexes 0.140 / 0.126 / 0.145, above the whole
  frame on every seed; within 8 px 34% (said 35%). **The grounding run was killed for memory** at 07:49 after
  seven hours (memoize: save intermediate results); fixed and restarted (`organic-grounded2`). gazs_s0 ended at
  130,000 (first words 0.47, two words 0.48, pages 0.20); its control tagn_s0 is still to come. **Artifact v5**
  (https://claude.ai/artifact/NLZYiDC19SLy3UHpsvRHwL): a card "An eye that follows what moves" (3 seeds), the
  held-thing card's word-cells caveat now says why, two references checked on Crossref (Abrams & Christ 2003; von
  Hofsten & Rosander 1997; one-line summaries from memory, for Kris to open).
- 2026-11-27 10:11 Health: all healthy. **Disk cleaned on Kris's word** (10:05; see the decisions): October brains
  and the unreferenced brains from before the 2026-11-10 review deleted, 133 GB → 277 GB free; the Commons video
  service had been given a floor at 09:22 (waits under 100 GB free). **The control for the relay finding** (VISION.md
  10:11; three seeds): at home (the photo benchmark) evening the relay again costs the delta reader 0.007–0.018; the
  gain is only on CORe50's middle fixation. The relay circuit learning on, first line (seed 0 whole frame): 0.357
  against 0.360 alone; seeds 1–2 run beside it (`organic-relaylearn-s1`, `-s2`). The grounding run: 10,000 of
  20,000 captions heard. tagr_s1 at 100,000.
- 2026-11-27 10:50 Health: all healthy (297 GB of disk free after the cleanup; ~/Downloads' partial files removed on
  Kris's word at 10:19). **The relay circuit learning along the stream, seed 0** (VISION.md 10:49): middle fixation
  0.348 → 0.371 (said 0.36–0.39), above the whole frame's 0.357; whole frame unchanged on three seeds. Still
  learning at 10,000 frames; about half of the batch's gain. Seeds 1–2 of the middle fixation due about 11:40.
  **Started:** the same question in the lives, source by source (`organic-relaysrc`; said before it runs).
