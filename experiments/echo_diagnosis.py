"""Why does a level answer text with the last word it heard? (docs/CONVERSATION.md)

For a brain (a level's pickle) and some utterances with the eyes closed: which route spoke (copy, own motor
habits, recall), what the word cells wanted to say and how sure they were, and what comes out with the copy
route off. Then the world's side: in the lessons the brain lived, how often IS the adult's reply the last word
it just heard, a word from somewhere else in the utterance, or a word that was not in it at all.
    python experiments/echo_diagnosis.py out/levels/words_s0_W1.pkl [out/words_s0.pkl ...]
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, words as split, tidy  # noqa: E402

SAY = ["hello!", "how are you?", "do you have a cat?", "do you say anything else than the last word?", "cat peculiar for you?",
       "cat onomatopeia", "what does the cow say?", "say ball", "ball is what i want you to say", "bye bye!", "do you want more milk?",
       "where is mommy?", "the dog is big", "dog cat ball"]
dark = np.zeros((64, 64, 3), np.float32)

cur = Curriculum(seed=0)
print("THE WORLD: where the adult's reply sits in what was just heard (training items)")
print(f"{'stage':<6}{'source':<22}{'items':>7}{'last word':>11}{'first':>7}{'elsewhere':>11}{'not heard':>11}{'eyes closed':>13}")
for stage in ["B", "W1", "W2", "P", "K"]:
    for src in sorted(cur.train[stage]):
        n = last = first = inside = absent = closed = 0
        for it in cur.train[stage][src][:3000]:
            q, a = split(tidy(it[1])), split(tidy(it[2][0]))
            if not q or not a:
                continue
            n += 1
            closed += cur.image(it[0]).max() == 0 if n <= 200 else 0
            if a == q[-len(a):]:
                last += 1
            elif a == q[: len(a)]:
                first += 1
            elif all(w in q for w in a):
                inside += 1
            else:
                absent += 1
        if n:
            print(f"{stage:<6}{src:<22}{n:>7}{last / n:>11.2f}{first / n:>7.2f}{inside / n:>11.2f}{absent / n:>11.2f}{closed / min(n, 200):>13.2f}")

for path in sys.argv[1:]:
    brain = pickle.load(open(path, "rb"))
    brain.learning = False
    w = brain.words
    routes = ["copy", "own", "recall"]
    print(f"\nTHE BRAIN: {path} (age {brain.age}, {len(w.units)} unit cells, {len(w.cues)} cues)")
    names = [brain.orch.output_neuron(r) for r in ("speak:copy", "speak:own", "speak:recall")]
    print("value of each route (mean over clusters that know it):", {r: round(float(brain.orch.value_of(n)), 3) if hasattr(brain.orch, "value_of") else "?" for r, n in zip(routes, names)})
    print(f"{'heard':<46}{'said':<14}{'route':<8}{'word cells would say':<24}{'sure':>6}   copy route off")
    for q in SAY:
        said = brain.live(dark, q, learn=False, explore=False)[0]
        route = "copy" if brain.copied else "recall" if brain.recalled is not None else "own"
        heard = brain.ear.units(q, False)
        rec, conf = w.recall(heard, None)
        gain = brain.copy_gain
        brain.copy_gain = 0.0
        off = brain.live(dark, q, learn=False, explore=False)[0]
        off_route = "copy" if brain.copied else "recall" if brain.recalled is not None else "own"
        brain.copy_gain = gain
        print(f"{q:<46}{said!r:<14}{route:<8}{rec!r:<24}{conf:>6.2f}   {off!r} ({off_route})")
    # what the word cells expect to hear after each of these, eyes closed: the content dendrite's top units and the context's
    for q in ["how are you?", "do you have a cat?", "what does the cow say?"]:
        heard = brain.ear.units(q, False)
        act, drive, ctx = w.expect(heard, None)
        top = lambda v: ", ".join(f"{w.units[i].decode('utf-8', 'replace')!r} {v[i]:.2f}" for i in np.argsort(-v)[:6])  # noqa: E731
        print(f"  after {q!r}: heard as {[u.decode('utf-8', 'replace') for u in heard]}\n    content: {top(drive)}\n    context: {top(ctx)}")
