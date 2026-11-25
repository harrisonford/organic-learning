"""A life through the curriculum (docs/CURRICULUM.md): stage by stage, gated by mastery.

    python experiments/curriculum_life.py out/cur_s0.jsonl
    SEED=0  CORTEX=path  EVERY=5000  STAGE_BUDGET=300000  TOTAL=2000000  RESUME=path.pkl
    MAX_CLUSTERS=24000  MAX_OUTPUTS=4096  MAX_GB=11  EAR_HABITUATION=0  NEURONS=8 (per cluster)
    OVERHEARD=0 (overheard sentences per moment, heard by the ear only)  HUB=1 HUB_CELLS=n (association hub)
    JUDGE=1 (reward = probability an adult accepts the answer, organic/judge.py; accepted own forms stand)
    WORDS=1 MAX_UNITS=n MAX_CUES=n (word cells made of the ear's own units, organic/words.py; replaces HUB)
    OBSERVE=0.5 (unasked comments on pictures per moment, organic/curriculum.py observation; needs SPEECH)
    TEST_N=300 (held-out items per stage: a sampling error of about 0.03; it was 60, about 0.06)  WORDING_N=100

Every EVERY moments the brain is tested on held-out items of every stage up to
the current one (and the next, to see what is coming). It is promoted when the
current stage's score reaches its pass mark and every earlier stage keeps at
least 80% of the score it had when passed. If a stage is not passed within
STAGE_BUDGET moments, the life ends with a verdict: this structure, with this
much experience, stops at that stage.
"""

import collections
import json
import os
import pickle
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import NAMES, OBSERVED, PASS, PERIODS, RETAIN, STAGES, Curriculum, grade, sound_sim, tidy  # noqa: E402
from organic.curriculum import words as curriculum_words  # noqa: E402
from organic.loop2 import LoopBrain2  # noqa: E402
from organic.peye import Parts  # noqa: E402

out_path = sys.argv[1] if len(sys.argv) > 1 else "out/cur_s0.jsonl"
SEED = int(os.environ.get("SEED", 0))
EVERY = int(os.environ.get("EVERY", 5000))
STAGE_BUDGET = int(os.environ.get("STAGE_BUDGET", 300000))
TOTAL = int(os.environ.get("TOTAL", 2000000))

if os.environ.get("RESUME"):
    brain = pickle.load(open(os.environ["RESUME"], "rb"))
    # a resumed brain can be given what it was born without (docs/VISION.md, 2026-11-17): eyes that fixate, match
    # neurons in the word cells, a first cortex layer that listens to the word cells
    if os.environ.get("LOOKING") == "1":
        brain.looking = True
    if os.environ.get("MATCH") == "1" and getattr(brain, "words", None) is not None and not brain.words.match:
        brain.words.enable_match()
    if os.environ.get("EPISODES") == "1" and getattr(brain, "words", None) is not None and brain.words.episodes is None:
        from organic.episodes import Episodes
        brain.words.episodes = Episodes(brain.words.visual_dim, brain.ear.width)
    if float(os.environ.get("TEACH", 0)) and getattr(brain, "words", None) is not None:
        brain.teach = float(os.environ["TEACH"])
        brain.words.feedback_dims = brain.cortex.layers[0].kinds * brain.cortex_grid ** 2
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
        brain = SpeakingBrain(parts, seed=SEED, size=64, hub=hub, hub_cells=int(os.environ.get("HUB_CELLS", 0)) or None,
                              words=os.environ.get("WORDS") == "1", max_units=int(os.environ.get("MAX_UNITS", 0)) or None,
                              max_cues=int(os.environ.get("MAX_CUES", 0)) or None, one_stream=os.environ.get("ONE_STREAM") == "1",
                              novel_copy=os.environ.get("NOVEL_COPY") == "1", imagery=os.environ.get("IMAGERY") == "1", sound=os.environ.get("SOUND") == "1", ear=os.environ.get("EAR_KIND"),
                              looking=os.environ.get("LOOKING") == "1", match=os.environ.get("MATCH") == "1", teach=float(os.environ.get("TEACH", 0)),
                              episodes=os.environ.get("EPISODES") == "1", **cfg)
    else:
        brain = LoopBrain2(parts, seed=SEED, size=64, **cfg)
    brain.curriculum = {"stage": os.environ.get("FIRST_STAGE", "BB"), "since": 0, "passed": {}, "promotions": []}
if os.environ.get("KEY") and getattr(getattr(brain, "words", None), "episodes", None) is not None:
    # KEY=looked: the episodic store keeps a moment under what the eye was on, not under the whole view (docs/VISION.md,
    # 2026-11-23; adopted for new lives the same day: Kris left the decision to the loop). Kept with the brain.
    brain.words.episodes.key = os.environ["KEY"]
if os.environ.get("GAZE"):  # GAZE=serial: the eye fixates one place at a time, by the eye commands (docs/VISION.md, 2026-11-23). Kept with the brain.
    brain.gaze = os.environ["GAZE"]
if os.environ.get("RPE") == "normalised":  # the reward's error against its own usual size (organic/loop2.py _credit; docs/VISION.md, 2026-11-24). Kept with the brain.
    brain.reward_spread = {}
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
    # the adult always says their own word after the child's (RECAST=0: an accepted answer's own form is taught, as
    # in every life before 2026-11-19; docs/JUDGING.md)
    brain.recast = os.environ.get("RECAST", "1") == "1"

if os.environ.get("NOVEL_DRIVE") == "1":
    brain.novel_drive = True  # the ear's novelty drives the copy route (docs/CONVERSATION.md, 2026-11-17)
if os.environ.get("INTAKE") == "1":
    brain.intake = True  # leisure talk: only the end of the parent's reply is learned from (docs/LEISURE.md, 2026-11-17)
if os.environ.get("FULL") == "1" and getattr(brain, "words", None) is not None:
    brain.words.plastic_by_error = False  # every correction of the content dendrite at full size (docs/VISION.md, 2026-11-17)
world = Curriculum(seed=0, world=int(os.environ.get("WORLD", 1)))  # WORLD=2: echo games worded by adults, social routines
PARROT = os.environ.get("PARROT") == "1"  # gate on the items a parrot of the last word heard gets wrong (docs/CONVERSATION.md)
world.rng.seed(1000003 * SEED + brain.age)  # a resumed life draws moments it has not drawn before
print(world.summary(), flush=True)
TEST_N = int(os.environ.get("TEST_N", 300))  # the ruler (docs/REVIEW_2026-11-10.md): 60 items gave +-0.06 per checkpoint
tests = {s: world.test_set(s, n=TEST_N) for s in STAGES}
state = brain.curriculum
if os.environ.get("FROM_LEVEL") == "1" and state["stage"] in state["passed"]:
    # a life continued from a frozen level (out/levels/): the level is the brain at the checkpoint it passed a
    # stage, written before it entered the next one; it enters it now, and babbles only as a living brain does
    state["promotions"].append((state["stage"], brain.age))
    state["stage"], state["since"] = STAGES[STAGES.index(state["stage"]) + 1], brain.age
    brain.babble_when_silent = False

# In sleep, moments of the fast store come back and the word cells live them again (organic/episodes.py, replay;
# docs/RESEARCH_MEMORY.md, 2026-11-18). REPLAY: how many come back for every moment of the day. REPLAY_FROM=P: only
# from that stage on (the experimenter's switch, to compare with a life that had none; a brain has no such switch).
REPLAY = float(os.environ.get("REPLAY", 0))
# How strict the adults are with the reward of a lived answer (docs/JUDGING.md, 2026-11-23; the tests are always strict):
# "strict": the answer must mean what the adult's did. "lenient": a good try counts in full ("woof" for a dog; the
# judge's NEAR). "ramp": kind while the child is small and stricter as it grows: a good try counts in full up to first
# words, half in the pages stage, not at all after. The adult's own answer is always what the learner is told.
REWARD = os.environ.get("REWARD", "strict")
KIND = {"BB": 1.0, "B": 1.0, "W1": 1.0, "W2": 0.5}
GATE = os.environ.get("GATE", "beyond")  # when the world moves on: "beyond" (the rigid grade beyond the parrot), "above" (a mark), "plateau" (no mark; see the promotion below)
SPAN = int(os.environ.get("PLATEAU_SPAN", 3))  # GATE=plateau: how many tests are averaged on each side when asking whether the curve still rises
PASS_ABOVE = float(os.environ.get("PASS_ABOVE", 0.25))  # provisional (docs/JUDGING.md): to be set with Kris
LAST_STAGE = os.environ.get("LAST_STAGE")  # a life that stays in this stage however it does there (the ablation lives, 2026-11-20)


def replay_on():
    ep = getattr(getattr(brain, "words", None), "episodes", None)
    if ep is None or not REPLAY or brain.replay:
        return
    ep.grow(int(os.environ.get("EPISODES_CAP", 6000)))
    ep.fade = float(os.environ.get("EPISODES_FADE", 6000))
    ep.returns = {}
    if os.environ.get("REPLAY_TAG") == "reward" and ep.tag is None:  # only moments at which reward came return in sleep (docs/VISION.md, 2026-11-24 night)
        ep.tag = np.zeros(ep.capacity, np.float32)
    brain.replay = REPLAY
    print(f"replay on at {brain.age}: the store holds {ep.n} moments of {ep.capacity}", flush=True)


if STAGES.index(state["stage"]) >= STAGES.index(os.environ.get("REPLAY_FROM", "BB")):
    replay_on()
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


def blind_floors(stage_):
    """What children who neither look nor listen get from the judge on a stage's test, item by item: always "yes",
    always "no", always the source's commonest answer, and the last word of the question said back
    (docs/JUDGING.md, 2026-11-19). A score means something only above these."""
    items = tests[stage_]
    srcs = [m["source"] for m in items]
    common = {src: collections.Counter(m["answers"][0] for m in items if m["source"] == src).most_common(1)[0][0] for src in set(srcs)}
    says = {"yes": lambda m: "yes", "no": lambda m: "no", "common": lambda m: common[m["source"]],
            "last": lambda m: (curriculum_words(tidy(m["question"])) or [""])[-1]}
    return {name: np.array(judge.accept([(stage_, m["question"], m["answers"], say(m)) for m in items],
                                        fallback=lambda i, say=say: grade(stage_, say(items[i]), items[i]["answers"]), sources=srcs))
            for name, say in says.items()}


FLOOR = {}  # by stage, worked out when the stage is first tested (the judge remembers its judgements)


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
        # what repeating the last word heard would score on the same items, and the brain's score where that fails
        par = [grade(s, (curriculum_words(tidy(m["question"])) or [""])[-1], m["answers"]) for m in tests[s]]
        last = [curriculum_words(tidy(said or "")) == curriculum_words(tidy(m["question"]))[-1:] for m, said in zip(tests[s], saids)]
        beyond = [x for x, p in zip(scores, par) if p < 0.5]
        res[s] = {"score": float(np.mean(scores)), "parrot": float(np.mean(par)), "beyond": float(np.mean(beyond)) if beyond else 0.0, "says_last": float(np.mean(last)),
                  "se": float(np.std(scores) / np.sqrt(max(len(scores), 1))), "exact": float(np.mean(exact)), "tick": float(np.mean(ticks)), "n": len(scores),
                  "by_source": {src: {"score": float(np.mean([x for x, _, _ in v])), "tick": float(np.mean([t for _, _, t in v])), "n": len(v),
                                      "said": [y for _, y, _ in v[:5]]} for src, v in by.items()}}
        if judge is not None:  # side by side with the local grade (the gate stays on the local grade for now)
            items = [(s, m["question"], m["answers"], said) for m, said in zip(tests[s], saids)]
            srcs = [m["source"] for m in tests[s]]
            if s not in FLOOR:
                FLOOR[s] = blind_floors(s)
            judged = judge.accept(items, fallback=lambda i: scores[i], sources=srcs)
            res[s]["judged"] = float(np.mean(judged))
            above, noise = [], []
            for src in res[s]["by_source"]:
                idx = np.array([i for i, x in enumerate(srcs) if x == src])
                mine = np.array(judged)[idx]
                # a question that saying its last word back answers shows nothing: the score is read on the others,
                # beside the best that always-yes, always-no or the commonest answer gets on those (the floor);
                # `above`: how far from that blind child towards an adult (docs/JUDGING.md, 2026-11-20)
                keep = idx[FLOOR[s]["last"][idx] < 0.5]
                entry = {"judged": float(np.mean(mine)), "asked": int(len(keep))}
                if len(keep) >= 10 and len(idx) >= 20:  # (a handful of items is no source of its own: the eight social routines)
                    floor = float(max(FLOOR[s][b][keep].mean() for b in ("yes", "no", "common")))
                    got = float(np.array(judged)[keep].mean())
                    entry.update(floor=floor, known=got, above=max(0.0, (got - floor) / max(1.0 - floor, 1e-6)))
                    above.append(entry["above"])
                    noise.append(float(np.array(judged)[keep].std() / np.sqrt(len(keep)) / max(1.0 - floor, 1e-6)))
                res[s]["by_source"][src].update(entry)
            res[s]["above"] = float(np.mean(above)) if above else 0.0
            res[s]["above_se"] = float(np.sqrt(np.sum(np.square(noise))) / len(noise)) if noise else 0.0  # how far this test's `above` may be off by chance
            # observed, not asked for: on yes/no questions, how much more often it says yes than yes is right
            # (two-year-olds say yes: +0.24 to +0.34 on this scale, Okanda & Itakura 2008)
            for src in res[s]["by_source"]:
                mine = [(tidy(said or "").split() or [""])[0] == "yes" for said, m in zip(saids, tests[s]) if m["source"] == src]
                want = [tidy(m["answers"][0]) for m in tests[s] if m["source"] == src]
                if want and set(want) <= {"yes", "no"} and len(set(want)) == 2:
                    res[s]["by_source"][src]["yes_bias"] = float(np.mean(mine) - np.mean([w == "yes" for w in want]))
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


OBSERVE = float(os.environ.get("OBSERVE", 0))  # unasked comments on pictures per moment (observation before interrogation)
owed_obs = 0.0


def observe(stage):
    global owed_obs
    owed_obs += OBSERVE
    while owed_obs >= 1:
        o = world.observation(stage)
        brain.observe(o["image"], o["says"])
        owed_obs -= 1


# leisure talk (docs/LEISURE.md): CHAT per moment, a parent says something and expects nothing; the brain says what it
# says, and the parent (a local chat model, organic/parent.py) answers that. No lesson, no grade, no reward.
CHAT = float(os.environ.get("CHAT", 0))
owed_chat = 0.0
if CHAT:
    import random as _random
    from organic.parent import Parent
    parent, chat_rng = Parent(), _random.Random(SEED + 77)
    chat_log = open(out_path.replace(".jsonl", ".chat.jsonl"), "a")
    social = [it[1] for s in ("W1", "W2") for it in world.train.get(s, {}).get("social", [])]
    DARK = np.zeros((64, 64, 3), np.float32)


def chat(stage):
    global owed_chat
    owed_chat += CHAT
    while owed_chat >= 1:
        owed_chat -= 1
        says, seen, about = None, DARK, None
        if world.by_name and chat_rng.random() < 0.5:  # half of the talk is about something the two are looking at
            about = chat_rng.choice(sorted(world.by_name))
            seen = world.image(chat_rng.choice(world.by_name[about]))
            says = parent.open(stage, chat_rng, topic=f"the {about} you are both looking at (use the word {about})")
        elif chat_rng.random() < 0.5 or not social:  # of the rest, half is new and half are the adults' written routines
            says = parent.open(stage, chat_rng, chat_rng.sample(social, 3) if len(social) >= 3 else ())
        says = says or (chat_rng.choice(social) if social else None)
        if not says:
            continue
        said, back = brain.chat(seen, says, lambda a, b: parent.answer(stage, a, b), babble=stage == "BB")
        chat_log.write(json.dumps({"age": brain.age, "stage": stage, "sees": about, "parent": says, "child": said, "back": back}) + "\n")
        chat_log.flush()


verdict = None
rewards = []
lived = {}  # what the life was given since the last checkpoint, by stage and source: the record shows the world it is in
while brain.age < TOTAL:
    stage = state["stage"]
    store = getattr(getattr(brain, "words", None), "episodes", None)
    if store is not None:
        store.mark_now = -1  # (a label for the record only: which stage's lesson a stored moment is from; -1: talk, comments)
    if OVERHEARD:
        overhear()
    if OBSERVE:
        observe(stage)
    if CHAT and hasattr(brain, "chat"):
        chat(stage)
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
        lived[m["stage"] + " " + m["source"]] = lived.get(m["stage"] + " " + m["source"], 0) + 1
        lived["with a picture"] = lived.get("with a picture", 0) + int(m["image"].max() > 0)
        if stage == "B" and hasattr(brain, "babble"):  # infancy: babbling between the words heard
            brain.babble(int(os.environ.get("BABBLE_PER", 3)))
        if judge is not None:
            def judged(said, m=m, kind=KIND.get(stage, 0.0) if REWARD == "ramp" else float(REWARD == "lenient")):
                ask = lambda near: judge.accept([(m["stage"], m["question"], m["answers"], said)], fallback=lambda i: grade(m["stage"], said, m["answers"]), sources=[m["source"]], near=near)[0]  # noqa: E731
                strict = ask(False)
                return strict + kind * max(0.0, ask(True) - strict) if kind and strict < 1.0 else strict
        if store is not None:
            store.mark_now = STAGES.index(m["stage"])
        _, tick, r, *_ = brain.live(m["image"], m["question"], m["answer"],
                                    grade=judged if judge is not None else (lambda said, m=m: grade(m["stage"], said, m["answers"])))
        rewards.append(r)
        if brain.age % EVERY:
            continue
    k = STAGES.index(stage)
    res = evaluate(STAGES[: k + 2])
    # with PARROT=1, from first words on a stage is passed (and held) on the items that repeating the last word does not answer
    # GATE=above (2026-11-19, docs/JUDGING.md): from first words on, the adult judge's score above what a child who
    # neither looks nor listens gets, and one pass mark for every stage: PASS_ABOVE of the way from that child to an adult
    # GATE=plateau (2026-11-21; Kris: a pass mark is a number of ours, not something the brain does): no mark at all.
    # The world moves on when the stage's own curve has stopped rising: the mean of its last SPAN tests is no higher
    # than the mean of the SPAN before them by more than the tests' own chance error. What the brain could do when
    # it left is recorded (the mean of the last SPAN tests), never asked for; earlier stages are held against that.
    asked = lambda s: GATE in ("above", "plateau") and s not in OBSERVED and "above" in res[s]  # noqa: E731
    mark = lambda s: res[s]["above"] if asked(s) else res[s]["beyond"] if PARROT and s not in ("BB", "B") and "beyond" in res[s] else res[s]["score"]  # noqa: E731
    need = PASS_ABOVE if asked(stage) else PASS[stage]
    cur = mark(stage)
    flat = False
    if GATE == "plateau" and asked(stage):
        curve = state.setdefault("curve", {}).setdefault(stage, [])
        curve.append((brain.age, cur, res[stage].get("above_se", 0.0)))
        if len(curve) >= 2 * SPAN:
            now, before = [c[1] for c in curve[-SPAN:]], [c[1] for c in curve[-2 * SPAN:-SPAN]]
            chance = float(np.sqrt(sum(c[2] ** 2 for c in curve[-2 * SPAN:])) / SPAN)  # the chance error of the difference of the two means
            flat = float(np.mean(now) - np.mean(before)) <= chance
            if flat:
                cur = float(np.mean(now))  # the level it leaves with
    # earlier stages must be held, except babbling and echo: those are observed, not asked for (organic/curriculum.py, OBSERVED)
    held = all(mark(s) >= RETAIN * state["passed"][s] for s in STAGES[:k] if s in state["passed"] and s in res and s not in OBSERVED)
    reached = flat if GATE == "plateau" and asked(stage) else cur >= need
    promoted = (reached or stage in PERIODS) and held and stage != LAST_STAGE  # (a period is lived through, not passed on a mark)
    if "W1" in res:
        res["W1"]["wording"] = wording_invariance()
    rec_drift = drift()
    rec = {"age": brain.age, "stage": stage, "drift": rec_drift, "at_stage": brain.age - state["since"], "measure": res, "lived_reward": float(np.mean(rewards[-EVERY:])),
           "clusters": int(brain.orch.C), "outputs": len(brain.orch.outputs), "promoted": promoted, "held_earlier": held,
           "observed": getattr(brain, "observed", 0), "words": brain.words.cells() if getattr(brain, "words", None) is not None else None,
           "judge": {"calls": judge.calls, "cached": judge.hits, "fallbacks": judge.fallbacks} if judge is not None else None,
           "memory_gb": round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9, 2),  # held now (the guard below); until 2026-11-18 the peak, which never falls
           "memory_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "seconds": round(time.time() - t0)}
    print(f"age {brain.age:>8} stage {stage:<2} ({brain.age - state['since']:>7} there) | "
          + " | ".join(f"{s} {v['score']:.2f}/{v['exact']:.2f}" + (f"/j{v['judged']:.2f}" if "judged" in v else "") + (f" (beyond parrot {v['beyond']:.2f}, parrot {v['parrot']:.2f}, says last {v['says_last']:.2f})" if "beyond" in v and s != "B" else "") for s, v in res.items()) + f" | clusters {brain.orch.C} | {'PROMOTED' if promoted else ''}", flush=True)
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
            if state["stage"] == os.environ.get("REPLAY_FROM"):
                replay_on()
    elif brain.age - state["since"] >= STAGE_BUDGET:
        verdict = f"stops at {stage} ({NAMES[stage]}): still rising at {cur:.2f} after {STAGE_BUDGET:,} moments" if GATE == "plateau" and asked(stage) and not flat else \
                  f"stops at {stage} ({NAMES[stage]}): score {cur:.2f} < {need} after {STAGE_BUDGET:,} moments" if cur < need and not flat else \
                  f"stops at {stage} ({NAMES[stage]}): passes it but cannot hold earlier stages"
    if (brain.age // EVERY) % int(os.environ.get("REFRESH_EVERY", 10)) == 0:  # new items written by the adults join in
        rng_state = world.rng.getstate()
        # the same world it was born into: until 2026-11-17 this line rebuilt world 1, so every life of world 2 or 3
        # (conv, edge, world3) lived in the old world from 50,000 moments on (docs/CRAFT.md)
        dice = world.prng.getstate()
        world = Curriculum(seed=0, world=int(os.environ.get("WORLD", 1)))
        world.rng.setstate(rng_state)
        world.prng.setstate(dice)
    rec["verdict"] = verdict
    ep = getattr(getattr(brain, "words", None), "episodes", None)
    if ep is not None and ep.mark is not None:  # the fast store: whose moments it holds, and which came back in sleep (how many, how surprising still)
        label = lambda k: STAGES[k] if k >= 0 else "talk"  # noqa: E731
        rec["episodes"] = {"held": int(ep.n), "from": {label(int(k)): int(c) for k, c in zip(*np.unique(ep.mark[: ep.n], return_counts=True))},
                           "came_back": {label(k): [c, round(t / c, 3)] for k, (c, t) in (ep.returns or {}).items()}}
        if ep.tag is not None:  # of each stage's stored moments, how many reward marked, and how much of it they hold together
            rec["episodes"]["rewarded"] = {label(int(k)): [int((ep.tag[: ep.n][ep.mark[: ep.n] == k] > 0).sum()), round(float(ep.tag[: ep.n][ep.mark[: ep.n] == k].sum()), 2)] for k in np.unique(ep.mark[: ep.n])}
        if ep.returns is not None:
            ep.returns = {}
    rec["lived"], lived = lived, {}
    log.write(json.dumps(rec) + "\n")
    log.flush()
    save()
    if verdict or rec["memory_gb"] > float(os.environ.get("MAX_GB", 1e9)):
        print("VERDICT:", verdict or f"stopped for memory at {rec['memory_gb']} GB", flush=True)
        break
