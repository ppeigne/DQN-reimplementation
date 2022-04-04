import numpy as np
import torch as T

from typing import Tuple

class Memory():
    def __init__(self, size:int, state_shape: Tuple[int, ...]) -> None:
        self.size = size
        self.current_states = T.zeros((size, *state_shape), dtype=T.float32)
        self.actions = T.zeros(size, dtype=T.int64)
        self.rewards = T.zeros(size, dtype=T.float32)
        self.next_states = T.zeros((size, *state_shape), dtype=T.float32)
        self.dones = T.zeros(size, dtype=T.bool)
        self.idx_last = 0

    def collect_experience(self, current_state, action, reward, next_state, done) -> None:
        index = self.idx_last % self.size

        self.current_states[index] = T.tensor(current_state)
        self.actions[index] = T.tensor(action)
        self.rewards[index] = T.tensor(reward)
        self.next_states[index] = T.tensor(next_state)
        self.dones[index] = T.tensor(done)
        self.idx_last += 1

    def get_sample(self, sample_size: int) -> Tuple[T.Tensor, ...]:
        if sample_size > self.idx_last:
            raise ValueError()
        
        past_frontier = min(self.size, self.idx_last)

        sample_idxs = np.random.choice(past_frontier, size=sample_size, replace=False)
        return (self.current_states[sample_idxs], self.actions[sample_idxs], self.rewards[sample_idxs],
                self.next_states[sample_idxs], self.dones[sample_idxs])