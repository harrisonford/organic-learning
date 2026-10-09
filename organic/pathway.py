"""Myelinated pathways between areas.

A pathway's conduction delay depends on its myelin. The receiving area
integrates its inputs within a time window before it commits to a response;
signals that arrive late count less. Myelination is activity dependent
(Gibson et al. 2014): pathways that carry a lot of signal get wrapped and
become faster, unused ones slowly lose myelin. So a young organism barely uses
its slow, unmyelinated vision-to-speech pathways and comes to rely on them as
they mature.
"""

import numpy as np


class Pathway:
    def __init__(self, name, length=1.0, myelin=0.05, deadline=2.0, growth=0.004, loss=0.0005):
        self.name = name
        self.length = length
        self.myelin = myelin
        self.deadline = deadline
        self.growth, self.loss = growth, loss

    @property
    def delay(self):
        return self.length / (0.2 + self.myelin)

    @property
    def gain(self):
        """Fraction of the signal that arrives inside the integration window."""
        return float(1.0 / (1.0 + np.exp((self.delay - self.deadline) / 0.3)))

    def carry(self, activity):
        """Activity-dependent myelination; activity in [0, 1]."""
        self.myelin += self.growth * activity * (1.0 - self.myelin) - self.loss * self.myelin
        self.myelin = float(np.clip(self.myelin, 0.0, 1.0))
