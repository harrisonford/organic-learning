"""A serial, word-led gaze in a small learner (docs/VISION.md, 2026-11-23): does it learn to look before it answers?

A learner built as the lives build theirs (the same parts and sizes), but small in time: it starts in first words
with an ear that has already heard speech (out/tmp/ear_first.pkl; the lives' learners carve theirs in the two
babbling stages), lives N1 moments of first words and then pages (world 4), and is graded by the letter (no judge,
no leisure talk, no unasked comments). Two learners a seed, the same moments in the same order:
  all at once   the word cells get the whole view and every fixated place before the first tick (the lives' looking)
  serial        the whole view only; each eye command of the gating policy brings what the fovea has where the eye
                lands; the map is biased by what stands out and by the heard words' pull; the adult's reply draws
                the eye once before the word cells learn (organic/speech.py, gaze = "serial"); the question draws
                the eye once before anything is said ("serial_noq": it does not, as first built)
  serial_shuf   the serial gaze with the words' pull shuffled over the places: the pull then says nothing about
                where the named thing is, and its learned weight should stay near nothing
Every EVERY moments, nothing learning: the grade on held-out first words and pages; the ticks before the answer
ends; on "is there a ...?" pages made here (so that it is known what is where): looks taken before answering, and,
where one candidate place holds the named thing and another does not, whether the first look went to it.
A small learner from a late start is not a life: this reads the mechanism, not the level a life would reach.
    SEEDS=0 N=6000 python experiments/gaze_life_probe.py
"""
import collections, os, pickle, re, resource, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402
from organic.peye import Parts  # noqa: E402
from organic.scenes import page, Scenes, plural  # noqa: E402
from organic.speech import SpeakingBrain  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
MODES = os.environ.get("MODES", "all,serial").split(",")
N, N1 = int(os.environ.get("N", 6000)), int(os.environ.get("N1", 2000))
EVERY, TESTS = int(os.environ.get("EVERY", 1000)), int(os.environ.get("TESTS", 100))
TILE, CELL = 52, 64


def held_names(cur):
    d = collections.defaultdict(list)
    for src in ("cifar:name", "commons:name"):
        for it in cur.test["W1"][src]:
            if cur.image(it[0]).max() > 0:
                d[it[2][0]].append(it[0])
    return d


def inside(places, side, tile):
    """Which places have the thing in quarter `tile` whole in the fovea (at least 0.9 of it)."""
    m = (CELL - TILE) // 2
    tr, tc = (tile // 2) * CELL + m, (tile % 2) * CELL + m
    return [max(0, min(r + side, tr + TILE) - max(r, tr)) * max(0, min(c + side, tc + TILE) - max(c, tc)) >= 0.9 * TILE * TILE for r, c in places]


def run(seed, mode):
    parts = Parts()
    parts.W = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "parts.npy"))
    parts.frozen = True
    brain = SpeakingBrain(parts, seed=seed, size=64, words=True, max_units=2000, max_cues=8192, novel_copy=True, looking=True, match=True, episodes=True,
                          reflex="contrast", adapt=True, sleep_cleanup=True, neurons_per_cluster=8,
                          cortex=pickle.load(open("data/visual_cortex2.pkl", "rb")), cortex_grid=4, cortex_layers=(0,))
    brain.ear = brain.words.ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
    brain.mouth.max_units = 300
    brain.orch.set_budget(int(os.environ.get("CLUSTERS", 1000)), 1024)  # the lives have 2,000; half here, for memory beside three lives
    brain.words.plastic_by_error = False
    brain.words.episodes.key = "looked"
    if mode.startswith("serial"):
        brain.gaze = "serial"
        brain.gaze_rate = float(os.environ.get("GAZE_RATE", brain.gaze_rate))
        brain.gaze_balance = os.environ.get("GAZE_BALANCE", brain.gaze_balance)  # "time" (default) or "learned"
        brain.look_on_question = mode != "serial_noq"  # serial_noq: the question does not draw the eye (the first version, 2026-11-23)
        if mode == "serial_shuf":  # the control of the map's learned weights: the words' pull lands on places at random
            pull, order = brain._pull, np.random.default_rng(seed + 31)
            brain._pull = lambda units: order.permutation(pull(units))
    world = Curriculum(seed=0, world=4)
    held = held_names(world)
    names = sorted(held)
    name_of = {r: w for w, refs in held.items() for r in refs}
    scenes = Scenes(held, world.image, 99, TILE, CELL)
    tests = {st: world.test_set(st, n=TESTS) for st in ("W1", "W2")}
    t0, ticks, looks = time.time(), [], []
    for n in range(1, N + 1):
        m = world.moment("W1" if n <= N1 else "W2")
        img = np.asarray(m["image"], np.float32)
        _, tick, *_ = brain.live(img, m["question"], m["answer"], grade=lambda said, m=m: grade(m["stage"], said, m["answers"]))
        ticks.append(tick)
        if brain._gazed is not None and img.shape[0] > 64:
            looks.append(len(brain._gazed["taken"]))
        if n % EVERY:
            continue
        out = [f"seed {seed} {mode:10} {n:6}"]
        for st, items in tests.items():
            g, tk = collections.defaultdict(list), []
            for m in items:
                said, tick, *_ = brain.live(np.asarray(m["image"], np.float32), m["question"], learn=False, explore=False)
                g["pages" if m["source"].startswith("page") else "single"].append(grade(st, tidy(said or ""), m["answers"]))
                tk.append(tick)
            out.append(f"{st} " + " ".join(f"{k} {np.mean(v):.2f}" for k, v in sorted(g.items())) + f" (ticks {np.mean(tk):.1f})")
        right, took, first, any_, moves = [], [], [], [], []
        for i in range(TESTS):
            tiles, shade, q, a = scenes.make("there", i)
            img = page([world.image(t) if t is not None else None for t in tiles], shade, TILE, CELL)
            said, tick, _, path, *_ = brain.live(img, q, learn=False, explore=False)
            right.append(tidy(said or "") == a)
            moves.append(len(path) - 1)
            gz = brain._gazed
            if gz is None or brain.gaze != "serial":
                continue
            took.append(len(gz["taken"]))
            asked = [w for w in names if re.search(rf"\b({re.escape(w)}|{re.escape(plural(w))})\b", q)]
            there = [k for k, t in enumerate(tiles) if t is not None and asked and name_of.get(t) == max(asked, key=len)]
            if there and gz["places"]:
                ins = np.any([inside(gz["places"], gz["side"], k) for k in there], axis=0)
                if ins.any() and (~ins).any():  # the order of looks can matter
                    any_.append(bool(gz["taken"]))
                    if gz["taken"]:
                        first.append(bool(ins[gz["places"].index(gz["taken"][0])]))
        out.append(f'"is there" pages right {np.mean(right):.2f}, eye moves {np.mean(moves):.2f}')
        if took:
            out.append(f"looks before answering {np.mean(took):.2f}; where it matters (n {len(any_)}): looked at all {np.mean(any_):.2f}, first look on the named thing {np.mean(first) if first else float('nan'):.2f} (n {len(first)})")
        if brain.gaze_gain is not None:
            out.append(f"the map's weights: stands out {brain.gaze_gain[0]:.3f}, the words' pull {brain.gaze_gain[1]:.3f}")
        out.append(f"living: ticks {np.mean(ticks):.1f}" + (f", looks a page (with the reply's) {np.mean(looks):.2f}" if looks else "") +
                   f" | clusters {int(brain.orch.C)} | {time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
        print(" | ".join(out), flush=True)
        ticks, looks = [], []


for seed in SEEDS:
    for mode in MODES:
        run(seed, mode)
