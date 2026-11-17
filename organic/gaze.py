"""Where the eye looks before anything is learned about it (docs/VISION.md, 2026-11-17).

A reflex, as in the superior colliculus: the eye is drawn to what stands out from its surround. No knowledge of
things, pages or questions.

  1. Adaptation: every place is seen relative to the scene's commonest colour (how far it differs from it). The
     average colour was tried first and finds the things on a page half as often (it is pulled by the things).
  2. Centre-surround: a window the size of the fovea's reach (half the view) stands out as far as its inside differs
     more than the ring around it does. Zero is a natural level: at or below it the window is no different from its
     surround, and nothing draws the eye there.
  3. The strongest window is fixated first; a place once fixated does not draw the eye again (inhibition of return).

What is fixated fills the view (the fovea's magnification). A learned gaze (the orchestrator's eye commands) can
later choose among these places; this only says which places there are.
"""
import numpy as np


def fixations(img, side=None, ring=None, most=6, apart=0.0):
    """Top-left corners (row, column) of the windows the eye fixates in a picture, the strongest first."""
    img = np.asarray(img, np.float32)
    H, W = img.shape[:2]
    side = side or H // 2
    ring = ring or side // 8
    step = side // 2  # the places the eye can point to: three by three
    energy = np.abs(img - np.median(img.reshape(-1, img.shape[2]), axis=0)).mean(axis=2)
    total = np.pad(energy, ring).cumsum(0).cumsum(1)
    total = np.pad(total, ((1, 0), (1, 0)))

    def area(r0, c0, r1, c1):  # sum of energy in rows r0:r1, columns c0:c1 of the padded picture (outside it: nothing)
        return total[r1, c1] - total[r0, c1] - total[r1, c0] + total[r0, c0]

    found = []
    for r in range(0, H - side + 1, step):
        for c in range(0, W - side + 1, step):
            inside = area(r + ring, c + ring, r + ring + side, c + ring + side)
            outside = area(r, c, r + side + 2 * ring, c + side + 2 * ring) - inside
            # the ring's area inside the picture only: beyond the picture's edge there is nothing to compare with
            r0, c0, r1, c1 = max(r - ring, 0), max(c - ring, 0), min(r + side + ring, H), min(c + side + ring, W)
            ring_area = (r1 - r0) * (c1 - c0) - side * side
            inner, outer = inside / (side * side), outside / max(ring_area, 1)
            # `apart`: the eye is drawn only to what stands apart, its surround at most this fraction as busy as itself
            # (0: anything busier than its surround; tried 2026-11-17 because fixating parts of a single thing costs its naming)
            if not apart or outer <= apart * inner:
                found.append((inner - outer, r, c))
    found.sort(reverse=True)
    out = []
    for s, r, c in found:
        if s <= 0 or len(out) >= most:
            break
        if all(abs(r - r2) >= side or abs(c - c2) >= side for r2, c2 in out):  # not where it has already looked
            out.append((r, c))
    return out
