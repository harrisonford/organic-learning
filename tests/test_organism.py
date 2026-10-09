import numpy as np

from organic.brain import Brain
from organic.cortex import Cortex
from organic.ear import SILENCE, Ear
from organic.eye import Eye
from organic.scenes import LOGIC_INTENTS, RELATION_INTENTS, SINGLE_INTENTS, episode


def test_ear_units_rebuild_the_exact_bytes():
    ear = Ear()
    rng = np.random.default_rng(0)
    for _ in range(50):
        e = episode(rng)
        for text in (e["question"], e["answer"]):
            assert b"".join(ear.units(text)) == text.encode()


def test_ear_starts_with_single_bytes():
    assert Ear().units("hello", learn=False) == [b"h", b"e", b"l", b"l", b"o"]


def test_eye_counts_objects():
    rng = np.random.default_rng(1)
    eye = Eye()
    for _ in range(40):
        e = episode(rng, "train", SINGLE_INTENTS + RELATION_INTENTS)
        assert len(eye.look(e["image"])) == len(e["objects"])


def test_new_inputs_do_not_change_old_clusters():
    cx = Cortex(dim=8, seed=0)
    s = np.ones(8, np.float32) / np.sqrt(8)
    target = cx.output_neuron("x")
    cx.learn(s, target, cx.output_group(""))
    before = cx._affinity(s)
    cx.grow_inputs(4)
    s2 = np.concatenate([s, np.ones(4, np.float32)])
    s2 /= np.linalg.norm(s2)
    assert np.allclose(cx._affinity(s2), before, atol=1e-5)


def test_learn_returns_forward_result():
    cx = Cortex(dim=4, seed=0)
    g = np.array([cx.output_neuron("a"), cx.output_neuron("b")])
    s = np.array([1, 0, 0, 0], np.float32)
    cx.learn(s, g[0], g)
    r = cx.learn(s, g[1], g)
    assert isinstance(r["forward"], dict) and r["forward"]["probs"] is not None


def test_brain_lives_and_answers_all_tasks():
    brain = Brain(sprouting=True)
    rng = np.random.default_rng(2)
    for _ in range(40):
        e = episode(rng, "train", SINGLE_INTENTS + LOGIC_INTENTS + RELATION_INTENTS)
        brain.live(e["image"], e["question"], e["answer"])
    brain._sprout()
    brain.live(None, "hello", "hi there")
    for intents in (SINGLE_INTENTS, LOGIC_INTENTS, RELATION_INTENTS):
        e = episode(rng, "test", intents)
        out, _ = brain.live(e["image"], e["question"], learn=False)
        assert isinstance(out, str)
    assert brain.known(SILENCE) is not None


def test_retina_streams_and_saccades():
    from organic.retina import Retina, develop_v1

    rng = np.random.default_rng(3)
    retina = Retina()
    e = episode(rng)
    stream = retina.view(e["image"], ticks=7)
    assert len(stream) == 7
    assert stream[0].parvo.sum() == 0  # parvocellular signal arrives a tick later
    assert stream[0].magno.sum() > 0  # magnocellular fires at fixation onset
    assert stream[0].fixation != stream[3].fixation  # it saccades
    develop_v1(retina, [e["image"]], waves=20, rng=rng)
    assert retina.view(e["image"], ticks=2)[1].v1.shape == (retina.rings, retina.angles, 16)
