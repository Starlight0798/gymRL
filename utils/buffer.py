from collections import deque
import torch
import numpy as np


class ReplayBuffer_on_policy:
    def __init__(self, cfg, capacity=5000):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=object)
        self.size = 0
        self.pointer = 0
        self.device = cfg.device
        
    def push(self, transitions):
        self.buffer[self.pointer] = transitions
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity
        
    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.size = 0
        self.pointer = 0
        
    def sample(self):
        return map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                          device=self.device), zip(*self.buffer[:self.size]))


class ReplayBuffer_off_policy:
    def __init__(self, cfg):
        self.buffer = np.empty(cfg.memory_capacity, dtype=object)
        self.size = 0
        self.pointer = 0
        self.capacity = cfg.memory_capacity
        self.batch_size = cfg.batch_size
        self.device = cfg.device

    def push(self, transitions):
        self.buffer[self.pointer] = transitions
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.size = 0
        self.pointer = 0

    def sample(self):
        batch_size = min(self.batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                             device=self.device), zip(*self.buffer[indices]))
        return samples