from collections import deque
import torch
import numpy as np

class ReplayBuffer_on_policy:
    def __init__(self, cfg, capacity=10000):
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
        states, actions, rewards, next_states, dones, log_probs, values, next_values = map(
            lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=self.device), 
            zip(*self.buffer[:self.size])
        )
        
        actions, rewards, dones, log_probs, values, next_values = actions.view(-1, 1).type(torch.long), \
            rewards.view(-1, 1), dones.view(-1, 1), log_probs.view(-1, 1), values.view(-1, 1), next_values.view(-1, 1)
        
        return states, actions, rewards, next_states, dones, log_probs, values, next_values


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
    
    
# numpy实现环形队列存储
class Queue:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = np.empty(buffer_size, dtype=object)
        self.index = 0
        self.filled = False

    def put(self, item):
        self.buffer[self.index] = item
        self.index = (self.index + 1) % self.buffer_size
        if self.index == 0:
            self.filled = True

    def sample(self):
        if not self.filled and self.index == 0:
            raise ValueError('Queue is empty!')
        max_index = self.buffer_size if self.filled else self.index
        idx = np.random.randint(0, max_index)
        return self.buffer[idx]

    def is_empty(self):
        return not self.filled and self.index == 0

    def is_full(self):
        return self.filled
    
    def size(self):
        return self.buffer_size if self.filled else self.index
    
    def capacity(self):
        return self.buffer_size