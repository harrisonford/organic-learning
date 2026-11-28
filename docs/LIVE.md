# The live page: a camera, a microphone and text streamed to the brain (2026-11-27)

Kris, 2026-11-27 16:00 and 21:30: a small app on SpacetimeDB that streams camera, text and sound to kid-a's brain in
real time and lets it do whatever it wants; "it doesn't matter if it's bad, I just want to see hints of something
forming"; SpacetimeDB so the page reacts well; hosting open ("we could host it ourselves but is there a better
quicker option like in spacetimedb itself?"); design decisions delegated while he sleeps. This file is the state
of that build: what runs, every decision and why, what was measured, what is not tested, what comes next.

## Where it is (21:51)

It runs. On the tailnet only: **port 9444 of the server's tailnet address, over https** (open it on a device that is on the
tailnet, press the button, allow camera and microphone). Code: `~/kid-a/live/` (committed there, not pushed);
the brain's side is `organic/live.py` and switches in `organic/stream.py`.

Tested end to end without a person: a synthetic page (a moving square, a tone) through the database to the brain
and back, and the real page in headless Chromium with a fake camera and microphone, over the tailnet address. **Not
tested: a real camera, a real voice, a phone's browser.** That is the first thing to find out in the morning.

## Hosting: the answer to Kris's question

- **The model cannot run inside SpacetimeDB.** A module is a sandboxed function (TypeScript on V8, or WebAssembly):
  no native libraries, no numpy, no state kept between calls, no file or network access in a reducer. SpacetimeDB's
  own guide for models is a "sidecar": a process that connects like any client. So the brain is a process *beside*
  the database wherever the database is.
- **The database could be on their cloud (Maincloud) in one command** (free tier: about 3 million calls a month).
  Not chosen: the brain would still run here, every frame and every 50 ms of sound would go to their cloud and come
  back (two extra trips), a camera's pictures would leave the house, and at 30 calls a second the free tier lasts a
  day of use.
- **Chosen: everything on arch-server.** SpacetimeDB 2.16.3 self-hosted (installed in user space, no root),
  the brain beside it, the page served from the same origin. A reducer call takes 3–4 ms round trip. The licence
  (BSL 1.1) allows one self-hosted instance. Cost: the server's CPU is shared with the lives and probes (below).
- If it ever goes public: Maincloud or a small VPS for the database and page, the brain staying here as a client
  that dials out. The code does not change, only the address.

## Shape

    page (TypeScript)  ──reducers──▶  SpacetimeDB module  ──subscription──▶  bridge (Node)  ──local socket──▶  brain
       ▲   camera 128×128 JPEG, 10 a second        event tables: frame_in, sound_in, text_in                    organic/live.py
       │   sound 16 kHz, 50 ms pieces              gaze_out, voice_out; brain_state (one kept row)              (stream brain,
       └──────────── subscription ◀──────────────────────────────────────────── reducers ◀──────────────────    10 ms clock)

Three user services: `kid-a-live-db` (SpacetimeDB, 127.0.0.1:3000, in memory; publishes the module when it
starts), `kid-a-live-brain` (Python), `kid-a-live-bridge` (Node). Tailscale serve, port 9444, tailnet only: `/` the
built page, `/v1` the database (the Funnel ports were not touched; checked after).

What the page shows: the camera with the eye's window over it and the trail of its jumps; what the cochlea handed
on (64 bands, 24 s); the mouth's 32 gestures in the vowel space, flashing when made and growing with habit; time
lived, counts, and how the brain keeps up with the clock. The brain's babble is played aloud.

## Decisions (mine, 2026-11-27 21:30–21:51; each can be undone)

1. **The brain is the stream brain (`organic/stream.py`) unchanged in what it does**, with the settings the probes
   ended on: both reflexes of the eye (what stands out and what moves; pursuit), eight stretches of bands, habits
   only from sounds heard. Not kid-a's turn-based brain, which has no clock. One brain (seed 0), born at 0 s on the
   page's first use, kept in `out/live/brain_s0.pkl` every five minutes and when it sleeps; it wakes from there.
2. **Sight works beside the clock** (`sight_later`): the visual cortex takes 50–100 ms a view on this machine while
   probes run; done inside the 10 ms step it made a step take 127 ms. Now the fixated window is handed to a thread
   and the sight population gets the answer when it is ready, about a tenth of a second late, which is what a
   visual cortex's latency is; windows that arrive while it is busy are never looked at. This is the first organ
   that is asynchronous in fact (docs/ASYNC.md, step (c)). With it a step takes 3–6 ms. Off by default: every
   probe's path is as before (events, link, sight and habits compared on a small run: the same).
3. **The camera frame is 128 pixels and the eye's window is its own 64**: the fovea gets the picture's real pixels
   instead of 32 enlarged (`view`). Motion is found at the view's scale.
4. **The sight–sound link is kept as a sum and a count** (one pass over 4 million synapses instead of three);
   the same numbers.
5. **A microphone is never silent**, and the stream brain called "sound" anything above digital zero. `room`: the
   ear keeps the quietest level of the last while (rises over 10 s) and the loudest (fades over 10 minutes) and a
   sound is what stands more than a tenth of the way from one to the other: the text-free ear's rule for quiet
   (a tenth of the loudest moment) measured above the background. **Provisional and set by hand** (three numbers);
   the right mechanism is the ear's own adaptation to the sound so far, measured tonight (`LIVE=2`, VISION.md
   19:52). Live page only.
6. **It sleeps when nothing comes in for 3 s**: its age is time lived with a world, and an idle page costs nothing.
7. **Streams are event tables** (broadcast, not kept) and **the database runs in memory**: SpacetimeDB's log is
   never compacted, and frames and sound would write about 9 GB a day. Nothing of the camera or the microphone is
   stored anywhere; the brain keeps only itself.
8. **The bridge is Node with the official client**, not Python: there is no maintained Python client, and the old
   JSON protocol cannot subscribe to event tables. Python and Node talk over a local socket.
9. **Private first**: tailnet only. The input tables are public inside the database (any connected page could read
   another's frames), which is acceptable for one viewer on a private network and not for anything public.
10. **Echo**: the browser's echo cancellation is asked for, so the brain's own babble from the loudspeaker should
    not come back through the microphone. Untested on real hardware; if it does come back, the brain hears itself
    twice (it already hears its own voice inside, as before).
11. **Root was used once**, for the Tailscale serve port (the same way port 9443 exists). No system package was
    installed; SpacetimeDB is in `~/.local`.

## Measured

- A 10 ms step: 3–6.6 ms beside three probes and a life (the cortex's look: 50–105 ms, 5–10 views a second);
  no sound lost; the brain's clock level with the wall clock.
- Reducer round trip: median 3–4 ms, worst 15–25 ms (local and through the tailnet address).
- About three eye jumps and two babbles a second, as on recorded video.
- Memory: database 260 MB, brain 140 MB asleep, bridge 30 MB.

## The word-learning ear on the page (2026-11-28 00:50)

The live brain now has the text-free ear and its link to sight (`give_ear` in `organic/stream.py`; the organ is
`organic/hear_stream.py`, checked to give the very same sound kinds as the offline ear). Its hearing cortex and
relay were at first developed on one SpokenCOCO reader before the life (`out/live/ear_born.pkl`); since
2026-11-28 both are born with the brain (decisions 17 and 18) and that file is no longer read. Its 2,048 sound kinds
start empty and are founded by what the microphone brings; every kind heard leaves a trace of about a second,
tied by Hebb's rule to what the eye holds. The page has a new panel, **what a sound reminds it of**: when a kind
is heard, the past view (of the last 300 fixations, kept in memory only, never on disk) most like what the eye
held with the kinds now sounding, and how far it stands out among them. In the first minutes that is chance.

More decisions:

12. **The kinds' ear hears the world without the brain's own babble** (its own voice is taken out, not mixed in):
    otherwise the babble, two a second, would found the kinds and deafen it to the speaker.
13. **The room rule for this ear, changed after a test with recorded speech as the microphone:** the first rule
    (a tenth of the way from the quietest to the loudest level, row by row) let through a fifth of the kinds in
    clean speech, because a row's level dips between syllables and the loudest moment is one voice's peak. Now a
    sound is what stands a quarter above the background on the ear's own smoothed loudness (`room_clear` 1.25, by
    hand): a room's hum alone founds nothing, clean speech keeps 78% of its kinds, quiet speech over a hum 2.5
    kinds a second. Still provisional.
14. **The relay is the reader's, not the speaker's.** A voice through another voice's relay is coded half as
    evenly (VISION.md 2026-11-27 18:34). The relay as a circuit that learns from the microphone would fit itself
    to the speaker (it gives 0.082 against 0.090 on the benchmark) but needs hours of sound to settle; not on the
    page yet. **Corrected 2026-11-28 03:45:** "hours" was a guess and is wrong. Measured: born random, the circuit
    is at rest after three minutes of sound, and on a new voice it is level with that voice's own circuit within
    a minute (VISION.md 03:45). It also forgets the voice it left as fast. What is not known yet is whether the
    sound kinds, founded on the relay's code, follow a relay that moves; two runs are measuring that, and the
    circuit goes on the page when they are read (`relay_learns`, `relay_later` in `organic/hear_stream.py` are
    built for it).
17. **The relay is a circuit born with the brain** (2026-11-28 04:34; replaces 14). The two runs: inside the
    stream brain a circuit born random with the first caption ends at 0.055 against 0.061 for a kept one and
    0.061 for the batch relay, not told apart with 1,000 test pairs (VISION.md 04:30): the kinds follow a relay
    born with them. So the page's ear now has nothing of any speaker in it but the hearing cortex under the
    relay (still developed on a SpokenCOCO reader). The circuit learns only from frames the ear holds to be
    sound, so that a room's hum does not become its world; a change every 3.3 s of sound, worked out beside the
    clock (about 0.1 s). The page shows how much sound it has learned from; it is at rest after about three
    minutes of speech, and follows a new speaker within one. **A second speaker, measured (VISION.md 05:27,
    06:30, 07:37; three pairs of readers, one seed each):** after 2,500 captions of one reader and 2,500 of
    another, the ear with the circuit has learned the second about as well as if born to it (0.95, 0.73 and
    1.02 of what that voice reaches alone; level by median rank in all three) and mostly still finds the
    first (1.00, 0.74, 0.94 of what it had at the change). With the relay it had before (fitted on one voice,
    fixed) the second voice ends at 0.41, 0.66 and 1.04 of alone: shut out when the voices are unlike, no
    harm when they are alike. What I wrote at 05:27 ("the first kept whole; the fixed relay shuts a second
    speaker out") was the first pair alone and too strong.

18. **The hearing cortex is born with the brain too** (2026-11-28 08:58; VISION.md 06:39, 07:51, 08:58). The last
    piece of the ear that was a SpokenCOCO reader's. `HearingLayer` in `organic/hear_stream.py`: the same 128
    kinds of 8 × 8 windows on the cochlea, with the correlations of its windows removed by a circuit of
    interneurons (the relay's rule, at the published rate for tens of cells) and its kinds learning by their
    own rule, each kind's rate falling with its own wins; one window a frame, from the frames the ear holds to
    be sound, learned from every 3.3 s of sound. Measured before it went on the page: in one pass with
    everything born, reader 2 ends at 0.086 against 0.079 with the cortex developed first (no cost; the
    cortex's kinds come to rest by themselves), and inside the stream brain 0.033 / 0.060 at 2,500 / 5,000
    captions against 0.039 / 0.061 for the ear kept from a finished life. Page test with recorded speech
    passes (594 kinds in 112 s; no sound lost). A row through the ear costs the same as before. So whoever
    speaks to the page first is the voice its hearing forms on. Not yet known: a real microphone and room;
    whether the cortex, slowed by then, serves a second speaker as well (runs started 08:59).

15. **Typed letters are a third stream** (2026-11-28 01:32; `give_letters`, `type` in `organic/stream.py`): the
    page sends every keystroke as it is typed; each letter excites the unit of the run of 3 and of 4 letters that
    ends with it (a fixed hash into 2,048 units, as the lives' byte ear codes runs; no word is cut); each unit
    leaves a trace of two seconds tied by Hebb's rule to what the eye holds; Enter or a two-second pause begins a
    new run. On the captions as written this code finds a caption's photograph as well as whole words do
    (0.093 against 0.088, VISION.md 01:29). The recall panel answers letters as it answers sounds. The two-second
    trace is by hand, inside the range measured for sound kinds with pauses.
16. **The recall panel compares against the views it keeps, less their own average** (not the life's average
    view), so that a view stands out for what is particular to it.

Tested again 04:31 with the circuit relay: 593 kinds founded in 112 s, the relay learned from 96 s of sound, a step
4.0 ms on a quiet machine, no sound lost, no errors.

Tested: the page in headless Chromium with 100 s of recorded speech as the microphone: 534 kinds founded, recall
pictures arriving, no errors; a step takes 8.5 ms with the ear (the margin to 10 ms is small on the loaded
server). What the link is worth is measured on the benchmark, not on the page: VISION.md 2026-11-28 00:36, 00:38.

## Limits, said plainly

- **It will not name or answer.** It ties sounds and typed letters to what it sees and shows what they call up;
  it does not say or write anything back but babble.
- What can form in a session: the map from gesture to sound, the babble's habits drifting toward vowels it hears
  (measured on synthetic adults over 600 s, VISION.md 2026-11-26), the sight–sound link (learned, never read out).
- The link's running mean weighs every pairing equally: after an hour a new pairing changes almost nothing.
- One brain for all pages; two pages at once would mix their cameras.
- The page's speed depends on what else the server runs. The probes are at Nice 15 and the brain is not.

## Next (in this order)

1. A real session (Kris): does the camera start on his device, is the gaze sensible, does the babble play, does
   the room rule hold in his room, does echo come back.
2. The word-learning ear inside the stream brain (docs/ASYNC.md, "The ear's way onto the clock"): live bands and
   the relay circuit are running tonight; then the kinds and their link to what the eye holds, and on the page:
   a sound kind heard calls up what was seen with it.
3. (Done 01:32: typed letters as a third stream.) Sounds and letters meeting: a word typed while it is said.
4. Private inputs and one stream a page before anything public.

## Operating it

    systemctl --user status kid-a-live-db kid-a-live-brain kid-a-live-bridge
    journalctl --user -u kid-a-live-brain -f
    cd ~/kid-a/live && npm run build && systemctl --user restart kid-a-live-bridge       # after changing page or bridge
    systemctl --user restart kid-a-live-db                                               # after changing the module
    systemctl --user stop kid-a-live-brain && rm ~/organic-learning/out/live/brain_s0.pkl && systemctl --user start kid-a-live-brain   # a newborn
    sudo tailscale serve --https=9444 off                                                # take the page down
