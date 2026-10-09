"""A raw eye: retina, saccades and a developmental V1. No object rules.

The organ knows nothing about objects, colors as words, shapes or places. It
turns light into a stream of neural activity over time, the way the primate
retina and early visual pathway do, and that stream is all the brain gets.

Retina (fixed wiring)
    Photoreceptors sit on a log-polar grid around the point of fixation:
    densely packed and sharp at the fovea, sparser and blurrier with
    eccentricity (each receptor pools light over an area that grows with its
    distance from the center), covering the whole visual field. Three cone
    types (L, M, S) respond to the light.

    Ganglion cells compare each receptor's center with its surround
    (difference of Gaussians, a surround three times wider):
      - luminance ON and OFF cells,
      - color-opponent cells: L vs M (red-green) and S vs L+M (blue-yellow),
        each ON and OFF.
    Two output pathways:
      - magnocellular: luminance only, transient (it responds to change, so
        it fires at the start of each fixation), and it arrives first;
      - parvocellular: luminance + color opponency, sustained, one tick later.

Saccades (fixed brainstem / colliculus rule)
    The eye starts at the center of the image. After each fixation it jumps
    to where the peripheral magnocellular response is strongest, except
    places it has recently looked at (inhibition of return). The proprioceptive
    sense of where the eye points goes along with the visual stream.

V1 (develops once, then frozen)
    Cortex is retinotopic, and the log-polar grid is roughly how the retina
    maps onto V1. Each V1 location looks at a small window of ganglion
    activity (neighbouring rings and angles). The same few kinds of simple
    cells tile the whole map. Their tuning self-organizes by competitive
    Hebbian learning, first on spontaneous retinal waves (as before eye
    opening), then on visual experience. Then it is frozen and becomes part
    of the organ.
"""

from dataclasses import dataclass, field

import numpy as np


def blur(img, sigma):
    """Separable Gaussian blur of an (H, W, C) image."""
    if sigma < 0.3:
        return img
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode="constant")
    tmp = sum(k[i] * pad[:, i : i + img.shape[1] + 0, :] for i in range(2 * r + 1))
    out = sum(k[i] * tmp[i : i + img.shape[0], :, :] for i in range(2 * r + 1))
    return out


def gaussian_code(value, centers, width):
    return np.exp(-0.5 * ((value - centers) / width) ** 2).astype(np.float32)


@dataclass
class Tick:
    tick: int
    fixation: tuple  # (x, y) in image pixels, where the eye points
    eye_position: np.ndarray  # proprioceptive population code
    magno: np.ndarray  # (rings, angles) luminance-change response, or zeros
    parvo: np.ndarray  # (rings, angles, 6) sustained ganglion response, or zeros
    v1: np.ndarray = field(default=None)  # (rings, angles, K) if V1 has developed


class Retina:
    def __init__(self, rings=14, angles=16, r0=0.7, growth=1.33, fixation_ticks=3, ior_radius=5.0, seed=0):
        self.rings, self.angles = rings, angles
        self.radii = r0 * growth ** np.arange(rings)
        self.theta = np.linspace(0, 2 * np.pi, angles, endpoint=False)
        # receptive field size grows with eccentricity (cortical magnification)
        self.rf = np.maximum(0.5, self.radii * (growth - 1) * 1.2)
        self.fixation_ticks = fixation_ticks
        self.ior_radius = ior_radius
        self.v1 = None  # set by develop_v1
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ receptors

    @staticmethod
    def cones(img):
        """L, M, S cone responses from RGB (overlapping spectral sensitivities)."""
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return np.stack([0.7 * r + 0.3 * g, 0.35 * r + 0.65 * g, 0.1 * g + 0.9 * b], axis=-1)

    def _pyramid(self, img):
        """The scene blurred at every receptive-field size the retina uses."""
        lms = self.cones(np.asarray(img, np.float32))
        sizes = sorted(set(np.round(np.concatenate([self.rf, 3 * self.rf]), 1)))
        return lms, {s: blur(lms, s) for s in sizes}

    def _sample(self, layer, xs, ys):
        """Bilinear sampling; light from outside the image is darkness."""
        H, W = layer.shape[:2]
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        out = np.zeros(xs.shape + (layer.shape[2],), np.float32)
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            v = np.zeros_like(out)
            v[ok] = layer[yi[ok], xi[ok]]
            out += w[..., None] * v
        return out

    def receptor_positions(self, fx, fy):
        xs = fx + self.radii[:, None] * np.cos(self.theta)[None, :]
        ys = fy + self.radii[:, None] * np.sin(self.theta)[None, :]
        return xs, ys

    # ------------------------------------------------------------- ganglion

    def ganglion(self, pyramid, fx, fy):
        """Center-surround responses at every receptor: (rings, angles, 6).

        Channels: lum ON, lum OFF, L-M ON, L-M OFF, S-(L+M) ON, S-(L+M) OFF.
        """
        xs, ys = self.receptor_positions(fx, fy)
        out = np.zeros((self.rings, self.angles, 6), np.float32)
        for i in range(self.rings):
            c = self._sample(pyramid[np.round(self.rf[i], 1)], xs[i], ys[i])
            s = self._sample(pyramid[np.round(3 * self.rf[i], 1)], xs[i], ys[i])
            lum = (c[:, 0] + c[:, 1]) / 2 - (s[:, 0] + s[:, 1]) / 2
            rg = c[:, 0] - s[:, 1]
            by = c[:, 2] - (s[:, 0] + s[:, 1]) / 2
            out[i] = np.stack([np.maximum(lum, 0), np.maximum(-lum, 0), np.maximum(rg, 0), np.maximum(-rg, 0), np.maximum(by, 0), np.maximum(-by, 0)], axis=-1)
        return out

    # ------------------------------------------------------------- saccades

    def _next_fixation(self, pyramid, shape, fx, fy, visited):
        """Jump to the strongest peripheral contrast not recently looked at."""
        g = self.ganglion(pyramid, fx, fy)
        energy = g[..., 0] + g[..., 1]
        xs, ys = self.receptor_positions(fx, fy)
        H, W = shape[:2]
        inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        energy = np.where(inside, energy, 0)
        for vx, vy in visited:
            d2 = (xs - vx) ** 2 + (ys - vy) ** 2
            energy = energy * (1 - np.exp(-0.5 * d2 / self.ior_radius**2))
        i, j = np.unravel_index(int(np.argmax(energy)), energy.shape)
        return float(xs[i, j]), float(ys[i, j])

    def eye_position(self, fx, fy, shape):
        H, W = shape[:2]
        return np.concatenate([gaussian_code(fx / W, np.linspace(0, 1, 7), 0.12), gaussian_code(fy / H, np.linspace(0, 1, 7), 0.12)])

    # ------------------------------------------------------------- the view

    def view(self, image, ticks=9):
        """Look at an image for a number of ticks; returns the activity stream."""
        image = np.asarray(image, np.float32)
        _, pyramid = self._pyramid(image)
        H, W = image.shape[:2]
        fx, fy = (W - 1) / 2, (H - 1) / 2
        visited, stream = [], []
        prev_lum = np.zeros((self.rings, self.angles), np.float32)
        since = 0
        for t in range(ticks):
            if since == self.fixation_ticks:
                visited.append((fx, fy))
                visited = visited[-4:]  # inhibition of return fades
                fx, fy = self._next_fixation(pyramid, image.shape, fx, fy, visited + [(fx, fy)])
                since = 0
                prev_lum = np.zeros_like(prev_lum)  # a saccade blanks the retina
            g = self.ganglion(pyramid, fx, fy)
            lum = g[..., 0] - g[..., 1]
            magno = np.abs(lum - prev_lum)  # transient: responds to change
            prev_lum = lum
            parvo = g if since >= 1 else np.zeros_like(g)  # sustained, arrives later
            tick = Tick(t, (fx, fy), self.eye_position(fx, fy, image.shape), magno, parvo)
            if self.v1 is not None:
                tick.v1 = self.v1.respond(parvo)
            stream.append(tick)
            since += 1
        return stream


class V1:
    """Simple cells tiling the retinotopic map, tuned by competitive Hebbian learning."""

    def __init__(self, kinds=16, window=3, channels=6, k_active=2, seed=0):
        self.kinds = kinds
        self.window = window
        self.channels = channels
        self.dim = window * window * (channels // 2)  # signed opponent channels
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 1, (kinds, self.dim)).astype(np.float32)
        self.W = w / np.linalg.norm(w, axis=1, keepdims=True)
        self.wins = np.zeros(kinds)
        self.k_active = k_active
        self.frozen = False

    def patches(self, g):
        """Every (ring, angle) window of ganglion activity; angles wrap around.

        ON and OFF cells are combined into signed opponent signals and each
        window loses its mean (local contrast normalization, as in retina and
        LGN), so what is left is the spatial pattern. Returns the normalized
        patterns and their contrast energy.
        """
        signed = g[..., 0::2] - g[..., 1::2]  # lum, L-M, S-(L+M)
        R, A, C = signed.shape
        h = self.window // 2
        padded = np.concatenate([signed[:, -h:], signed, signed[:, :h]], axis=1)
        padded = np.pad(padded, ((h, h), (0, 0), (0, 0)))
        out = np.empty((R, A, self.window, self.window, C), np.float32)
        for i in range(R):
            for j in range(A):
                out[i, j] = padded[i : i + self.window, j : j + self.window]
        out = out - out.mean(axis=(2, 3), keepdims=True)
        out = out.reshape(R, A, -1)
        energy = np.linalg.norm(out, axis=-1)
        return out / (energy[..., None] + 1e-6), energy

    def respond(self, g):
        """Rectified, sparse (k-winners per location) simple-cell responses,
        scaled by local contrast."""
        p, energy = self.patches(g)
        y = np.maximum(p @ self.W.T, 0) * energy[..., None]  # (R, A, K)
        if self.k_active < self.kinds:
            kth = np.partition(y, -self.k_active, axis=-1)[..., -self.k_active][..., None]
            y = np.where(y >= kth, y, 0)
        return y

    def learn(self, g, rate=0.02, n_samples=24, rng=None):
        """Competitive Hebbian step: the best-matching cell moves toward the input."""
        if self.frozen:
            return
        rng = rng or np.random.default_rng()
        p, energy = self.patches(g)
        p, norms = p.reshape(-1, self.dim), energy.ravel()
        # Hebbian change needs real activity: only well-driven windows teach
        active = np.nonzero(norms > 0.3 * norms.max())[0] if norms.max() > 1e-3 else []
        if len(active) == 0:
            return
        for idx in rng.choice(active, min(n_samples, len(active)), replace=False):
            x = p[idx]
            # conscience: frequent winners are handicapped so every cell finds a niche
            score = self.W @ x - 0.3 * (self.wins / (self.wins.mean() + 1e-8) - 1)
            c = int(np.argmax(score))
            self.wins[c] += 1
            w = self.W[c] + rate * (x - self.W[c])
            self.W[c] = w / np.linalg.norm(w)


def retinal_wave(rng, size=40):
    """Spontaneous activity before eye opening: a drifting blob of correlated firing."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size, 3), np.float32)
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0, size, 2)
        a, b = rng.uniform(2, 10, 2)
        ang = rng.uniform(0, np.pi)
        dx, dy = xx - cx, yy - cy
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = -dx * np.sin(ang) + dy * np.cos(ang)
        blob = np.exp(-0.5 * ((u / a) ** 2 + (v / b) ** 2))
        img += blob[..., None] * rng.uniform(0.3, 1.0)  # spontaneous waves drive all cones alike
    return np.clip(img, 0, 1)


def develop_v1(retina, experience, waves=1500, rng=None, kinds=16, window=3):
    """Development: spontaneous waves, then visual experience, then freeze."""
    rng = rng or np.random.default_rng(0)
    v1 = V1(kinds=kinds, window=window, seed=int(rng.integers(1 << 30)))
    for n in range(waves):
        img = retinal_wave(rng)
        _, pyr = retina._pyramid(img)
        fx, fy = rng.uniform(5, 35, 2)
        v1.learn(retina.ganglion(pyr, fx, fy), rate=0.05 * (1 - n / waves) + 0.005, rng=rng)
    for n, img in enumerate(experience):
        for tick in retina.view(img, ticks=6):
            if tick.parvo.any():
                v1.learn(tick.parvo, rate=0.01, rng=rng)
    v1.frozen = True
    retina.v1 = v1
    return v1
