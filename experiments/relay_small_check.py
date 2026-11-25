"""The relay circuit on a small made-up input (256 cells, variances falling as k^-1.25): does it come to its resting
point at a rate where one picture's step is small, and what does it do where the step is large?"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relay_local_probe as P
D, rng = 256, np.random.default_rng(0)
lam = np.arange(1, D + 1) ** -1.25
lam = lam / lam.mean()
Q = np.linalg.qr(rng.standard_normal((D, D)))[0].astype(np.float32)
draw = lambda n: ((rng.standard_normal((n, D)) * np.sqrt(lam)).astype(np.float32) @ Q.T) / np.sqrt(D)  # unit length on average
pick = [1, 3, 10, 30, 60, 100, 200]
print("input ", " ".join(f"{lam[k-1]:.3g}" for k in pick), "| rest:", " ".join(f"{min(lam[k-1],1):.3g}" for k in pick))
test = draw(4000)
for rate in (1e-3, 1e-2, 3e-2):
    P.RATE = rate
    c = P.Circuit(D, np.random.default_rng(1))
    for n in range(0, 20000, 50):
        c.learn(draw(50))
        if n + 50 in (1000, 5000, 20000):
            Y = c.code(test); Y = Y - Y.mean(0)
            print(f"rate {rate:g} after {n+50:5}:", " ".join(f"{v:.3g}" for v in (Y @ Q[:, [k-1 for k in pick]]).var(0)), f"| squared length {float((Y*Y).sum(1).mean()):.1f}, rate x that {rate*float((Y*Y).sum(1).mean()):.3f}")
