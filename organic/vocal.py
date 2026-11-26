"""A vocal tract to babble with: sound from gestures, by the source-filter account of speech (docs/ASYNC.md; queue E 32).

The stream brain's mouth has been making symbols. A child's mouth makes sound, and babbling teaches it what its own
gestures sound like (the DIVA account: Guenther, Ghosh & Tourville 2006). The simplest real vocal tract: a glottal
source (a pulse train at the voice's pitch, F0) shaped by the resonances of the tract (formants F1, F2, and a fixed F3),
each a two-pole resonator in cascade, as in Klatt's (1980) synthesizer (much simplified), and the lips' radiation. A gesture is a place in the vowel space (F1, F2):
the tongue's height and front-back position, roughly. What it sounds like is not known to the brain: it hears it.

    tract = VocalTract(f0=280)                       # a child's pitch
    samples = tract.say(f1=700, f2=1200, ms=200)     # 16 kHz mono floats
"""
import numpy as np

RATE = 16000


def resonator(x, f, bw):
    """Klatt's (1980) two-pole resonator at frequency f (Hz), bandwidth bw (Hz), unit gain at 0 Hz: in cascade the
    formants then give the natural spectral envelope, peaks at the formants."""
    from scipy.signal import lfilter
    C = -np.exp(-2 * np.pi * bw / RATE)
    B = 2 * np.exp(-np.pi * bw / RATE) * np.cos(2 * np.pi * f / RATE)
    A = 1 - B - C
    return lfilter([A], [1, -B, -C], x).astype(np.float32)


class VocalTract:
    def __init__(self, f0=280.0, f3=2900.0, jitter=0.02, seed=0):
        self.f0, self.f3, self.jitter, self.rng = f0, f3, jitter, np.random.default_rng(seed)

    def say(self, f1, f2, ms=200):
        n = int(RATE * ms / 1000)
        src = np.zeros(n, np.float32)
        t = 0.0
        while t < n:  # glottal pulses, a little irregular
            src[int(t)] = 1.0
            t += RATE / (self.f0 * (1 + self.jitter * self.rng.standard_normal()))
        src = resonator(src, 0.0, 100.0)  # the glottal pulse's own fall with frequency (a low-pass at the source)
        y = resonator(resonator(resonator(src, f1, 60 + 0.05 * f1), f2, 80 + 0.04 * f2), self.f3, 150)
        y = np.diff(y, prepend=0.0)  # radiation at the lips: a rise of 6 dB an octave
        env = np.sin(np.linspace(0, np.pi, n)) ** 0.5  # onset and offset
        y = y * env
        return (0.3 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)
