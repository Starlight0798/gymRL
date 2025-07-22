import torch
import numpy as np

class ReplayBuffer_on_policy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = []
        self.samples = None
        
    def store(self, transitions):
        assert self.samples is None, 'Need to clear the buffer before storing new transitions.'
        self.buffer.append(transitions)
        
    def clear(self):
        self.buffer = []
        self.samples = None
        
    def size(self):
        return len(self.buffer)
    
    def compute_advantage(self, rewards, dones, dw, values, next_values):
        with torch.no_grad():
            td_error = rewards + self.cfg.gamma * next_values * (1 - dw) - values
            td_error = td_error.cpu().detach().numpy()
            dones = dones.cpu().detach().numpy()
            adv, gae = [], 0.0
            for delta, d in zip(td_error[::-1], dones[::-1]):
                gae = self.cfg.gamma * self.cfg.lamda * gae * (1 - d) + delta
                adv.append(gae)
            adv.reverse()
            adv = torch.tensor(np.array(adv), device=self.cfg.device, dtype=torch.float32).view(-1, 1)
            v_target = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
        return adv, v_target
        
    def sample(self):
        if self.samples is None:
            states, actions, rewards, dones, dw, log_probs, values, next_values = map(
                lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=self.cfg.device), 
                zip(*self.buffer)
            )
            
            actions, rewards, dones, dw, log_probs, values, next_values = actions.view(-1, 1).type(torch.long), \
                rewards.view(-1, 1), dones.view(-1, 1), dw.view(-1, 1), log_probs.view(-1, 1), values.view(-1, 1), next_values.view(-1, 1)
            
            adv, v_target = self.compute_advantage(rewards, dones, dw, values, next_values)
            self.samples = states, actions, log_probs, adv, v_target
        
        return self.samples


class ReplayBuffer_on_policy_v2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clear()

    def clear(self):
        episode_max_steps = self.cfg.max_steps
        self.buffer = {
            's': np.zeros([self.cfg.batch_size, episode_max_steps] + list(self.cfg.state_shape), dtype=np.float32),
            'a': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.int64),
            'a_logprob': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'r': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'd': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'dw': np.ones([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'v': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'v_': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'active': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.int8),
        }
        self.size = np.zeros(self.cfg.batch_size, dtype=int)
        self.episode_num = 0

    def store(self, transitions):
        s, a, r, d, dw, a_logprob, v, v_ = transitions
        self.buffer['s'][self.episode_num, self.size[self.episode_num]] = s
        self.buffer['a'][self.episode_num, self.size[self.episode_num]] = a
        self.buffer['a_logprob'][self.episode_num, self.size[self.episode_num]] = a_logprob
        self.buffer['r'][self.episode_num, self.size[self.episode_num]] = r
        self.buffer['d'][self.episode_num, self.size[self.episode_num]] = d
        self.buffer['dw'][self.episode_num, self.size[self.episode_num]] = dw
        self.buffer['v'][self.episode_num, self.size[self.episode_num]] = v
        self.buffer['v_'][self.episode_num, self.size[self.episode_num]] = v_
        self.buffer['active'][self.episode_num, self.size[self.episode_num]] = 1
        self.size[self.episode_num] += 1

    def next_episode(self):
        self.episode_num += 1

    def sample(self):
        max_episode_len = self.size.max()
        return (
            torch.tensor(self.buffer['s'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['a'][:, :max_episode_len], dtype=torch.long, device=self.cfg.device),
            torch.tensor(self.buffer['a_logprob'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['r'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['d'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['dw'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['v'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['v_'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['active'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
        )


class ReplayBuffer_off_policy:
    def __init__(self, cfg):
        self.buffer = np.empty(cfg.memory_capacity, dtype=object)
        self.is_full = False
        self.pointer = 0
        self.capacity = cfg.memory_capacity
        self.batch_size = cfg.batch_size
        self.device = cfg.device

    def store(self, transitions):
        self.buffer[self.pointer] = transitions
        self.pointer = (self.pointer + 1) % self.capacity
        if self.pointer == 0:
            self.is_full = True

    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.pointer = 0
        self.is_full = False

    def sample(self):
        batch_size = min(self.batch_size, self.size())
        indices = np.random.choice(self.size(), batch_size, replace=False)
        samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                             device=self.device), zip(*self.buffer[indices]))
        return samples
    
    def size(self):
        if self.is_full:
            return self.capacity
        return self.pointer
    
    
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