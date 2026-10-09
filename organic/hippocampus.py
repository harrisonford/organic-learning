"""The hippocampus: fast, one-shot memory of episodes, replayed during sleep.

Cortex learns slowly and is easily disturbed by whatever happened last. The
hippocampus keeps a store of recent experiences and, during sleep, replays
them to cortex in shuffled order (complementary learning systems,
McClelland, McNaughton & O'Reilly 1995). Old memories are forgotten when the
store is full, so cortex has to consolidate what matters.
"""

import numpy as np


class Hippocampus:
    def __init__(self, capacity=1500, seed=11):
        self.capacity = capacity
        self.episodes = []
        self.rng = np.random.default_rng(seed)

    def store(self, episode):
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(int(self.rng.integers(len(self.episodes) // 2)))  # older ones fade first

    def replay(self, n):
        if not self.episodes:
            return []
        idx = self.rng.choice(len(self.episodes), min(n, len(self.episodes)), replace=False)
        return [self.episodes[i] for i in idx]
