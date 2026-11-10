"""A life through the curriculum (docs/CURRICULUM.md): stage by stage, gated by mastery.

    python experiments/curriculum_life.py out/cur_s0.jsonl
    SEED=0  CORTEX=path  EVERY=5000  STAGE_BUDGET=300000  TOTAL=2000000  RESUME=path.pkl
    MAX_CLUSTERS=24000  MAX_OUTPUTS=4096  MAX_GB=11  EAR_HABITUATION=0  NEURONS=8 (per cluster)
    OVERHEARD=0 (overheard sentences per moment, heard by the ear only)  HUB=1 HUB_CELLS=n (association hub)
    JUDGE=1 (reward = probability an adult accepts the answer, organic/judge.py; accepted own forms stand)
    TEST_N=300 (held-out items per stage: a sampling error of about 0.03; it was 60, about 0.06)  WORDING_N=100

Every EVERY moments the brain is tested on held-out items of every stage up to
the current one (and the next, to see what is coming). It is promoted when the
current stage's score reaches its pass mark and every earlier stage keeps at
least 80% of the score it had when passed. If a stage is not passed within
STAGE_BUDGET moments, the life ends with a verdict: this structure, with this
much experience, stops at that stage.
"""

import json
import os
import pickle
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import NAMES, PASS, RETAIN, STAGES, Curriculum, grade, sound_sim, tidy  # noqa: E402
from organic.loop2 import LoopBrain2  # noqa: E402
from organic.peye import Parts  # noqa: E402

out_path = sys.argv[1] if len(sys.argv) > 1 else "out/cur_s0.jsonl"
SEED = int(os.environ.get("SEED", 0))
EVERY = int(os.environ.get("EVERY", 5000))
STAGE_BUDGET = int(os.environ.get("STAGE_BUDGET", 300000))
TOTAL = int(os.environ.get("TOTAL", 2000000))

if os.environ.get("RESUME"):
    brain = pickle.load(open(os.environ["RESUME"], "rb"))
else:
    parts = Parts()
    parts.W = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "parts.npy"))
    parts.frozen = True
    cfg = {"reflex": "contrast", "adapt": True, "sleep_cleanup": True, "neurons_per_cluster": int(os.environ.get("NEURONS", 8))}
    if os.environ.get("CORTEX"):
        cfg["cortex"] = pickle.load(open(os.environ["CORTEX"], "rb"))
        cfg["cortex_grid"] = int(os.environ.get("CORTEX_GRID", 1))
        cfg["cortex_layers"] = tuple(int(x) for x in os.environ.get("CORTEX_LAYERS", "1,2").split(","))
    if os.environ.get("SPEECH"):  # the speaking brain (organic/speech.py)
        from organic.speech import SpeakingBrain
        hub = os.environ.get("HUB") == "1"  # the association hub (organic/hub.py)
        brain = SpeakingBrain(parts, seed=SEED, size=64, hub=hub, hub_cells=int(os.environ.get("HUB_CELLS", 0)) or None, **cfg)
    else:
        brain = LoopBrain2(parts, seed=SEED, size=64, **cfg)
    brain.curriculum = {"stage": os.environ.get("FIRST_STAGE", "BB"), "since": 0, "passed": {}, "promotions": []}
if hasattr(brain, "mouth") and os.environ.get("MOTOR_UNITS"):  # the motor repertoire's size (each unit widens every cluster)
    brain.mouth.max_units = int(os.environ["MOTOR_UNITS"])
if os.environ.get("EAR_HABITUATION") == "0":
    brain.ear.habituation = False
if os.environ.get("MAX_CLUSTERS") or os.environ.get("MAX_OUTPUTS"):
    brain.orch.set_budget(int(os.environ.get("MAX_CLUSTERS", 0)) or None, int(os.environ.get("MAX_OUTPUTS", 0)) or None)

# the adults judge answers as people do (organic/judge.py, docs/JUDGING.md): JUDGE=1 makes the reward the
# probability an adult accepts the answer, and lets an accepted answer's own form stand (no correction)
judge = None
if os.environ.get("JUDGE") == "1":
    from organic.judge import Judge
    judge = Judge()
    brain.own_forms = True

world = Curriculum(seed=0)
world.rng.seed(1000003 * SEED + brain.age)  # a resumed life draws moments it has not drawn before
print(world.summary(), flush=True)
TEST_N = int(os.environ.get("TEST_N", 300))  # the ruler (docs/REVIEW_2026-11-10.md): 60 items gave +-0.06 per checkpoint
tests = {s: world.test_set(s, n=TEST_N) for s in STAGES}
state = brain.curriculum
log = open(out_path, "a")
t0 = time.time()


HEARD = [world.moment(st)["question"] for st in ("B", "W1", "P") for _ in range(70)]  # speech around the infant (for babble)


def babble_score(n=60):
    """How much its babble sounds like the speech it hears (sound likeness to the closest heard utterance)."""
    bab = [brain.babble(explore=True) for _ in range(n)]
    return float(np.mean([max(sound_sim(b, h) for h in HEARD) for b in bab])), bab


def wording_invariance(n=int(os.environ.get("WORDING_N", 100))):
    """Generalization, not score: the same test picture asked three different ways. Does the brain give
    the same answer each time (it understood the request, not the sound of one frame), and the right one?"""
    import zlib
    from organic.world import WORLD_DIR
    path = os.path.join(WORLD_DIR, "naming_frames.json")
    frames = json.load(open(path))["frames"] if os.path.exists(path) else ["what is this?"]
    pics = [m for m in tests["W1"] if m["source"] in ("cifar:name", "commons:name")]
    pics = pics[:: max(1, len(pics) // n)][:n]  # spread over both sources
    same = right = 0
    for k, m in enumerate(pics):
        asks = [frames[(zlib.crc32(m["question"].encode()) + 17 * j) % len(frames)] for j in range(3)]
        said = [tidy(brain.live(m["image"], q, learn=False, explore=False)[0]) for q in asks]
        same += len(set(said)) == 1 and said[0] != ""
        right += all(x in [tidy(a) for a in m["answers"]] for x in said)
    return {"same_answer": same / max(len(pics), 1), "right_every_time": right / max(len(pics), 1), "n": len(pics)}


PROBE_PICS = [m["image"] for st in ("W1", "W2", "P") for m in tests[st][:14]][:40]
PROBE_SAID = [m["question"] for st in ("B", "W1", "W2", "P", "K") for m in tests[st][:8]][:40]
PROBE_FILE = out_path.replace(".jsonl", ".probe.npz")


def organ_codes():
    """What each organ makes of a fixed probe set (pictures, utterances): its code, to compare over the life."""
    codes = {}
    eye = brain.eye
    if hasattr(eye, "parts"):
        centre = len(eye.sensor.targets) // 2
        codes["v1"] = np.array([eye.parts.encode(eye.sensor.views(im, centre)[0]) for im in PROBE_PICS])
    if getattr(brain, "cortex", None) is not None:
        codes["visual_cortex"] = np.array([np.concatenate(brain.cortex.code(im, grid=brain.cortex_grid)) for im in PROBE_PICS])
    codes["ear"] = np.array([brain.ear.question(brain.ear.units(q, learn=False), adapt=False) for q in PROBE_SAID])
    if hasattr(brain, "mouth"):
        codes["mouth_sound_map"] = brain.mouth.amap[:97].copy()
    return codes


def drift():
    """Organ drift since the previous checkpoint: 1 - mean cosine of each probe item's code then and now
    (0 = the organ's code did not change; settling shows as drift falling over the life). docs/PLASTICITY.md"""
    now = organ_codes()
    out = {}
    if os.path.exists(PROBE_FILE):
        before = np.load(PROBE_FILE)
        for k, a in now.items():
            if k in before and before[k].shape == a.shape:
                b = before[k]
                live = (np.linalg.norm(a, axis=1) > 0) & (np.linalg.norm(b, axis=1) > 0)  # rows never used (zero) say nothing
                cos = (a * b).sum(1)[live] / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)[live]
                out[k] = round(float(1 - cos.mean()), 5) if live.any() else 0.0
    np.savez(PROBE_FILE + ".tmp.npz", **now)
    os.replace(PROBE_FILE + ".tmp.npz", PROBE_FILE)
    return out


def evaluate(stages):
    res = {}
    for s in stages:
        if s == "BB":
            if hasattr(brain, "babble"):
                sc, bab = babble_score()
                res[s] = {"score": sc, "exact": 0.0, "tick": 0.0, "n": len(bab), "by_source": {"babble": {"score": sc, "n": len(bab), "said": bab[:5]}}}
            continue
        scores, ticks, exact, by, saids = [], [], [], {}, []
        for m in tests[s]:
            said, tick, *_ = brain.live(m["image"], m["question"], learn=False, explore=False)
            saids.append(said)
            scores.append(grade(s, said, m["answers"]))
            exact.append(tidy(said) in [tidy(a) for a in m["answers"]])
            ticks.append(tick)
            by.setdefault(m["source"], []).append((scores[-1], said, tick))
        res[s] = {"score": float(np.mean(scores)), "se": float(np.std(scores) / np.sqrt(max(len(scores), 1))), "exact": float(np.mean(exact)), "tick": float(np.mean(ticks)), "n": len(scores),
                  "by_source": {src: {"score": float(np.mean([x for x, _, _ in v])), "tick": float(np.mean([t for _, _, t in v])), "n": len(v),
                                      "said": [y for _, y, _ in v[:5]]} for src, v in by.items()}}
        if judge is not None:  # side by side with the local grade (the gate stays on the local grade for now)
            items = [(s, m["question"], m["answers"], said) for m, said in zip(tests[s], saids)]
            judged = judge.accept(items, fallback=lambda i: scores[i])
            res[s]["judged"] = float(np.mean(judged))
            for src in res[s]["by_source"]:
                res[s]["by_source"][src]["judged"] = float(np.mean([p for p, m in zip(judged, tests[s]) if m["source"] == src]))
    return res


def save():
    pkl = out_path.replace(".jsonl", ".pkl")
    with open(pkl + ".tmp", "wb") as f:
        pickle.dump(brain, f)
    os.replace(pkl + ".tmp", pkl)
    with open(pkl + ".ear.tmp", "wb") as f:  # the ear alone (small), for probes that cannot load the whole brain
        pickle.dump(brain.ear, f)
    os.replace(pkl + ".ear.tmp", pkl.replace(".pkl", ".ear.pkl"))


CLUSTER_ARRAYS = ("Z", "Win", "Wout", "myelin", "stability", "usage", "born", "uid", "reach", "patience", "last_active")


def save_frozen(path):
    """The release candidate, written from the brain already in memory (no second copy):
    the cortex arrays are swapped for views of the living clusters only (pickling a view
    stores just its rows), babbling when silent is switched on, then everything is put back.
    Returns its size: parameters (as kid-a counts them) and the RAM it needs to run."""
    o = brain.orch
    keep = {a: getattr(o, a) for a in CLUSTER_ARRAYS}
    old = (o.cap, o.max_clusters, getattr(brain, "babble_when_silent", False))
    C = max(o.C, 1)
    try:
        for a in CLUSTER_ARRAYS:
            setattr(o, a, keep[a][:C])
        o.cap, o.max_clusters = C, None
        if hasattr(brain, "babble"):
            brain.babble_when_silent = True
        with open(path + ".tmp", "wb") as f:
            pickle.dump(brain, f)
        os.replace(path + ".tmp", path)
        reach = int(o.reach[: o.C].sum())
        params = reach * (1 + o.n) + o.C * o.n * len(o.outputs) + 3 * o.C
        return {"parameters": int(params), "memory_bytes": int(os.path.getsize(path))}  # its arrays are nearly all of the file
    finally:
        for a in CLUSTER_ARRAYS:
            setattr(o, a, keep[a])
        o.cap, o.max_clusters, brain.babble_when_silent = old


# overheard speech (Simple English Wikipedia, experiments/world/simplewiki_speech.py): sentences heard by the ear
# between moments, not addressed to the brain and not answered (docs/EAR_SEGMENTATION.md); OVERHEARD per moment
OVERHEARD = float(os.environ.get("OVERHEARD", 0))
if OVERHEARD:
    from organic.world import DATA
    overheard = open(os.path.join(DATA, "simplewiki", "sentences.txt"))
    overheard.seek(int(1000003 * SEED + brain.age) % os.path.getsize(overheard.name))
    overheard.readline()
owed = 0.0


def overhear():
    global owed
    owed += OVERHEARD
    while owed >= 1:
        line = overheard.readline() or (overheard.seek(0) or overheard.readline())
        brain.ear.units(line.strip())
        owed -= 1


verdict = None
rewards = []
while brain.age < TOTAL:
    stage = state["stage"]
    if OVERHEARD:
        overhear()
    if stage == "BB":  # infancy before words: hearing the speech around it, babbling back
        m = world.moment("B" if brain.age % 3 == 0 else "W1" if brain.age % 3 == 1 else "P")
        heard = brain.ear.units(m["question"])
        brain.mirror(m["question"])  # covert imitation of the speech it hears
        brain.heard_recent = 0.9 * brain.heard_recent + 0.1 * brain.ear.question(heard)
        brain.babble(int(os.environ.get("BABBLE_PER", 3)))
        brain.age += 1
        rewards.append(0.0)
        if brain.age % EVERY:
            continue
    else:
        m = world.moment(stage)
        if stage == "B" and hasattr(brain, "babble"):  # infancy: babbling between the words heard
            brain.babble(int(os.environ.get("BABBLE_PER", 3)))
        if judge is not None:
            def judged(said, m=m):
                return judge.accept([(m["stage"], m["question"], m["answers"], said)], fallback=lambda i: grade(m["stage"], said, m["answers"]))[0]
        _, tick, r, *_ = brain.live(m["image"], m["question"], m["answer"],
                                    grade=judged if judge is not None else (lambda said, m=m: grade(m["stage"], said, m["answers"])))
        rewards.append(r)
        if brain.age % EVERY:
            continue
    k = STAGES.index(stage)
    res = evaluate(STAGES[: k + 2])
    cur = res[stage]["score"]
    # earlier stages must be held, except babbling, which is meant to fade as words come
    held = all(res[s]["score"] >= RETAIN * state["passed"][s] for s in STAGES[:k] if s in state["passed"] and s in res and s != "BB")
    promoted = cur >= PASS[stage] and held
    if "W1" in res:
        res["W1"]["wording"] = wording_invariance()
    rec_drift = drift()
    rec = {"age": brain.age, "stage": stage, "drift": rec_drift, "at_stage": brain.age - state["since"], "measure": res, "lived_reward": float(np.mean(rewards[-EVERY:])),
           "clusters": int(brain.orch.C), "outputs": len(brain.orch.outputs), "promoted": promoted, "held_earlier": held,
           "judge": {"calls": judge.calls, "cached": judge.hits, "fallbacks": judge.fallbacks} if judge is not None else None,
           "memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "seconds": round(time.time() - t0)}
    print(f"age {brain.age:>8} stage {stage:<2} ({brain.age - state['since']:>7} there) | "
          + " | ".join(f"{s} {v['score']:.2f}/{v['exact']:.2f}" + (f"/j{v['judged']:.2f}" if "judged" in v else "") for s, v in res.items()) + f" | clusters {brain.orch.C} | {'PROMOTED' if promoted else ''}", flush=True)
    if promoted:
        state["passed"][stage] = cur
        # a release candidate: the brain as it was when it passed this stage
        os.makedirs("out/levels", exist_ok=True)
        tag = os.path.basename(out_path).replace(".jsonl", "")
        sizes = save_frozen(f"out/levels/{tag}_{stage}.pkl")
        json.dump({"stage": stage, "name": NAMES[stage], "age": brain.age, "measure": res, "passed": dict(state["passed"]), "clusters": int(brain.orch.C),
                   "brain": type(brain).__name__, "frozen": True, **sizes}, open(f"out/levels/{tag}_{stage}.json", "w"), indent=1)
        state["promotions"].append((stage, brain.age))
        if k + 1 == len(STAGES):
            verdict = "passed every stage"
        else:
            state["stage"], state["since"] = STAGES[k + 1], brain.age
    elif brain.age - state["since"] >= STAGE_BUDGET:
        verdict = f"stops at {stage} ({NAMES[stage]}): score {cur:.2f} < {PASS[stage]} after {STAGE_BUDGET:,} moments" if cur < PASS[stage] else \
                  f"stops at {stage} ({NAMES[stage]}): passes it but cannot hold earlier stages"
    if (brain.age // EVERY) % int(os.environ.get("REFRESH_EVERY", 10)) == 0:  # new items written by the adults join in
        rng_state = world.rng.getstate()
        world = Curriculum(seed=0)
        world.rng.setstate(rng_state)
    rec["verdict"] = verdict
    log.write(json.dumps(rec) + "\n")
    log.flush()
    save()
    if verdict or rec["memory_gb"] > float(os.environ.get("MAX_GB", 1e9)):
        print("VERDICT:", verdict or f"stopped for memory at {rec['memory_gb']} GB", flush=True)
        break
