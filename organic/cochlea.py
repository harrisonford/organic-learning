"""A cochlea: sound as the inner ear hands it on (queue E 32, docs/LOOP.md; 2026-11-26).

The ear of every life so far hears text as bytes. A brain hears pressure waves, and before any learning the cochlea
has already split them by frequency: each place along the basilar membrane answers a band of frequencies, the
bands widen with frequency (equivalent rectangular bandwidths, Glasberg & Moore 1990), and the outer hair cells
compress loud and soft into a narrow range. This is the fixed front end, as the retina's patch whitening is for the
eye; what is learned starts after it.

    bands      64, spaced evenly on the ERB-rate scale from 80 Hz to 7,600 Hz, each a triangle on that scale
    frames     one every 10 ms, from 25 ms of sound (a Hann window)
    loudness   the energy in each band to the power 0.3 (a compressive nonlinearity of the kind outer hair cells give;
               Stevens' power law for loudness has about this exponent)

    from organic.cochlea import Cochlea
    gram = Cochlea().hear(samples)      # samples: mono, 16 kHz, floats in [-1, 1] -> (frames, 64) float32
"""
import numpy as np

RATE, HOP, WINDOW, BANDS, LOW, HIGH = 16000, 160, 400, 64, 80.0, 7600.0


def erb_rate(f):
    return 21.4 * np.log10(1.0 + 0.00437 * f)


def from_erb_rate(e):
    return (10 ** (e / 21.4) - 1.0) / 0.00437


class Cochlea:
    def __init__(self, bands=BANDS, low=LOW, high=HIGH, n_fft=512, exponent=0.3):
        self.n_fft, self.exponent = n_fft, exponent
        edges = from_erb_rate(np.linspace(erb_rate(low), erb_rate(high), bands + 2))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / RATE)
        e, fe = erb_rate(freqs), erb_rate(edges)
        W = np.zeros((bands, len(freqs)), np.float32)
        for b in range(bands):
            lo, mid, hi = fe[b], fe[b + 1], fe[b + 2]
            W[b] = np.clip(np.minimum((e - lo) / (mid - lo), (hi - e) / (hi - mid)), 0, None)
        self.W, self.centres, self.window = W, edges[1:-1], np.hanning(WINDOW).astype(np.float32)

    def hear(self, samples):
        x = np.asarray(samples, np.float32)
        if len(x) < WINDOW:
            return np.zeros((0, len(self.W)), np.float32)
        n = 1 + (len(x) - WINDOW) // HOP
        idx = np.arange(WINDOW)[None, :] + HOP * np.arange(n)[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * self.window, self.n_fft)) ** 2
        return ((spec @ self.W.T) ** self.exponent).astype(np.float32)
