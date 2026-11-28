"""A childhood for the brain of the live page (2026-11-28 11:12; Kris: "i want the trained model even if its bad").

The page's brain (organic/live.py LiveBrain: the stream brain on a 10 ms clock, with an ear born whole, typed
letters and a babbling mouth) is born when the page first wakes it, and knows nothing. Here the very same brain,
built by the very same code, lives a childhood before it is hosted: one reader's spoken captions, each heard once
on the clock, 10 ms at a time, while the photograph it describes fills the eye's view; and, as it is spoken, the
caption's letters arrive as typed (a letter about every 70 ms, as a line of subtitles would). Half a second of
silence, the photograph still in view, and the next. Nothing is told to it about words; the transcripts' letters
are an input stream like the sound, as on the page.

Kept every LEVELS captions and at the end, each a later state of the one brain (HOME/brain_s{SEED}_level{n}.pkl, and
the newest as HOME/brain_s{SEED}.pkl, which organic/live.py can host as it is), with the photographs it has seen and
their codes (HOME/childhood.npz), so that the page can show which photograph of its childhood a sound or typed
letters call up.

What differs from life on the page, and why: the photograph is 64 pixels across (all we hold), the size of the
eye's own window, so it is given to the eye whole in place of a camera frame the eye would move over (its
saccades go on over an enlarged copy, as bookkeeping only); the silence between captions is true silence.

Read at each level, nothing learning, on 1,000 photographs never seen with captions never heard: the caption among
them for its photograph from the typed letters (all 1,000) and from the sound (SOUND_TEST of them at a level, all
at the end), R@1, 5, 10 and the median rank.

    READER=top2 SEED=0 python experiments/live_childhood.py
"""
import io, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.hear_stream import unit  # noqa: E402
from organic.live import LiveBrain  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
READER, SEED, TEST, MAXU = os.environ.get("READER", "top2"), int(os.environ.get("SEED", 0)), int(os.environ.get("TEST", 1000)), int(os.environ.get("MAXU", 0))
HOME, SOUND_TEST, SILENCE = os.environ.get("HOME_DIR", "out/live_raised"), int(os.environ.get("SOUND_TEST", 300)), int(os.environ.get("SILENCE", 50))
LEVELS = [int(x) for x in os.environ.get("LEVELS", "1000,2500,5000,7500").split(",")]
t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731

rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1]
by_image = {}
for u in utts:
    if u["speaker"] == who:
        by_image.setdefault(u["image"], []).append(u)
images = sorted(by_image)
images = [images[i] for i in rng.permutation(len(images))]
test_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[:TEST]]
life_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[TEST:]][: MAXU or None]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
raw = lambda u: np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)  # noqa: E731
row_of = {i: r for r, i in enumerate(json.load(open(os.path.expanduser("~/datasets/coco_thumbs/train2017_ids.json"))))}
thumbs = np.load(os.path.expanduser("~/datasets/coco_thumbs/train2017_64.npy"), mmap_mode="r")
thumb = lambda u: np.asarray(thumbs[row_of[u["image"]]])  # noqa: E731
say(f"reader {READER}: a childhood of {len(life_u)} captions ({sum(u['rows'] for u in life_u) / 360000:.1f} hours of speech), each once with its photograph; {len(test_u)} held out")


class Held:
    """In place of the cortex beside the clock: the eye holds the photograph whole (its code is worked out before)."""
    code, seen, took, stop = None, None, [], False

    def __call__(self, window):
        pass


os.makedirs(HOME, exist_ok=True)
for f in (f"brain_s{SEED}.pkl",):  # a childhood starts from birth
    if os.path.exists(os.path.join(HOME, f)):
        os.remove(os.path.join(HOME, f))
live = LiveBrain(seed=SEED, home=HOME, emit=lambda e: None)
live.sight.stop = True
b, held = live.brain, Held()
b.sight_later = held
ear = b.kinds_ear


def codes(us):
    return np.stack([live.cortex.whole(thumb(u).astype(np.float32) / 255, 4) for u in us]).astype(np.float32)


V_life, V_test = codes(life_u), codes(test_u)
say("the photographs' codes")
QUIET = np.zeros(64, np.float32)


def hear(u, v):
    """A caption on the clock with its photograph in view and its letters arriving as it is spoken; then silence."""
    G, text, events = raw(u), u["text"].lower().encode(), []
    held.code = v
    at = {int(k * len(G) / max(1, len(text))): k for k in range(len(text))}
    big = np.kron(thumb(u).astype(np.float32) / 255, np.ones((2, 2, 1), np.float32))  # for the eye's reflexes only
    for j, row in enumerate(G):
        e = b.step(big if j == 0 else None, row)
        if j in at:
            b.type(text[at[j]], e)
        events += e
    b.type(10)
    for _ in range(SILENCE):
        events += b.step(None, QUIET)
    return events


def keep(name):
    b.cortex, b.hearing, log, b.sight_later = None, None, b.log, None
    b.log = []
    try:
        pickle.dump(b, open(os.path.join(HOME, name + ".tmp"), "wb"))
        os.replace(os.path.join(HOME, name + ".tmp"), os.path.join(HOME, name))
    finally:
        b.cortex, b.hearing, b.log, b.sight_later = live.cortex, live.hearing, log, held


rec = lambda r: " ".join(f"R@{k} {np.mean(r < k):.3f}" for k in (1, 5, 10))  # noqa: E731


def ranks(R, V, mean):
    S = unit(R) @ unit(V - mean).T
    return (S > np.diag(S)[None, :]).sum(0), (S > np.diag(S)[:, None]).sum(1)


def read(n, sound_n):
    """Nothing learning. Letters: a caption's letters typed at once, what they call up. Sound: the caption heard."""
    L = b.letters
    if L.link is not None:
        kept, typed = L.trace.copy(), (b.typed, b.typed_at)
        R, none = np.zeros((len(test_u), V_test.shape[1]), np.float32), 0
        for i, u in enumerate(test_u):
            L.trace[:] = 0
            b.typed = b" "
            for byte in u["text"].lower().encode():
                b.type(byte)
            r = L.recall()
            none += r is None
            if r is not None:
                R[i] = r
        L.trace[:], (b.typed, b.typed_at) = kept, typed
        r_img, r_cap = ranks(R, V_test, (L.sight_sum / L.sight_time).astype(np.float32))
        say(f"after {n} captions, typed letters: photograph -> its caption among {len(test_u)}: {rec(r_img)}; caption -> its photograph: {rec(r_cap)}; median rank {np.median(r_img) + 1:.0f} / {np.median(r_cap) + 1:.0f} (called up nothing: {none})")
    us, V = test_u[:sound_n], V_test[:sound_n]
    state = (ear.mu.copy(), ear.var.copy(), ear.top, b.kind_trace.copy(), b.floor, b.top)
    ear.pause()
    ear.learning = b.kind_learn = False
    R, none = np.zeros((len(us), V.shape[1]), np.float32), 0
    for i, u in enumerate(us):
        ear.mu, ear.var, ear.top = state[0].copy(), state[1].copy(), state[2]
        b.kind_trace[:] = 0
        total, events = np.zeros_like(b.kind_trace), []
        for row in raw(u):
            b._hear_kinds(row, True, events)
            total += b.kind_trace
        for what, k in ear.pause():
            b.kind_trace[k + (ear.k23 if what == "triple" else 0)] += 1.0
        total += b.kind_trace * SILENCE / 2
        r = b.recall(total)
        none += r is None
        if r is not None:
            R[i] = r
    ear.mu, ear.var, ear.top = state[0], state[1], state[2]
    b.kind_trace[:], b.floor, b.top = state[3], state[4], state[5]
    ear.learning = b.kind_learn = True
    if b.sight_time > 0:
        r_img, r_cap = ranks(R, V, (b.sight_sum / b.sight_time).astype(np.float32))
        say(f"after {n} captions, the sound: photograph -> its caption among {len(us)}: {rec(r_img)}; caption -> its photograph: {rec(r_cap)}; median rank {np.median(r_img) + 1:.0f} / {np.median(r_cap) + 1:.0f}"
            f" (called up nothing: {none}; sound units founded {ear.pairs.founded} + {ear.triples.founded}; babbles so far {babbles}; its ear held {held_sound / max(1, rows_heard):.2f} of the sound's rows to be sound)")


def level(n):
    keep(f"brain_s{SEED}.pkl")
    if n in LEVELS or n == len(life_u):
        keep(f"brain_s{SEED}_level{n}.pkl")
    jpegs = []
    from PIL import Image
    for u in life_u[:n]:
        out = io.BytesIO()
        Image.fromarray(thumb(u)).save(out, "JPEG", quality=80)
        jpegs.append(np.frombuffer(out.getvalue(), np.uint8))
    np.savez(os.path.join(HOME, "childhood.tmp.npz"), codes=V_life[:n].astype(np.float16), jpeg=np.concatenate(jpegs), ends=np.cumsum([len(j) for j in jpegs]))
    os.replace(os.path.join(HOME, "childhood.tmp.npz"), os.path.join(HOME, "childhood.npz"))
    say(f"kept at {n} captions: the brain is {b.t / 3600:.2f} hours old")


babbles = rows_heard = held_sound = 0
for n, u in enumerate(life_u, 1):
    ev = hear(u, V_life[n - 1])
    babbles += sum(e[0] == "sound" for e in ev)
    rows_heard += u["rows"]
    held_sound += u["rows"] - int(np.sum(ear.quiet[-u["rows"]:])) if len(ear.quiet) >= u["rows"] else 0
    if n in LEVELS or n == len(life_u):
        read(n, SOUND_TEST if n < len(life_u) else len(test_u))
        level(n)
    elif n % 250 == 0:
        keep(f"brain_s{SEED}.pkl")
        say(f"{n} captions, {b.t / 3600:.2f} hours lived, {(time.time() - t0) / n:.2f} s a caption")
