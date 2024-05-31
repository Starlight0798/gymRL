import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from utils.model import *
from utils.buffer import ReplayBuffer_off_policy as ReplayBuffer
from utils.runner import *
from torch.optim.lr_scheduler import CosineAnnealingLR

class Config(BasicConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.env_name = 'CartPole-v1'
        self.algo_name = 'NDQN'
        self.train_eps = 500
        self.lr_start = 1e-3
        self.lr_end = 1e-5
        self.batch_size = 128
        self.preload_size = 256
        self.memory_capacity = 10000
        self.target_update = 400

class DQNnet(nn.Module):
    def __init__(self, cfg):
        super(DQNnet, self).__init__()
        self.fc = MLP([cfg.n_states, 64], last_act=True)
        self.fc_a = MLP([64, 32, cfg.n_actions], linear=NoisyLinear)
        self.fc_v = MLP([64, 32, 1], linear=NoisyLinear)

    def forward(self, s):
        x = self.fc(s)
        a = self.fc_a(x)
        v = self.fc_v(x)
        q = v + (a - torch.mean(a, dim=-1, keepdim=True))
        return q

class DQN(ModelLoader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.memory = ReplayBuffer(cfg)
        self.net = DQNnet(cfg).to(cfg.device)
        self.target_net = DQNnet(cfg).to(cfg.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr_start)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.train_eps // 4, eta_min=cfg.lr_end)
        self.cfg = cfg
        self.learn_step = 0
        self.predict_step = 0
        self.lr = cfg.lr_start

    @torch.no_grad()
    def choose_action(self, state):
        self.predict_step += 1
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float32)
        action = self.net(state).argmax(dim=-1).item()
        return action

    @torch.no_grad()
    def evaluate(self, state):
        return self.choose_action(state)

    def update(self):
        if self.memory.size < self.cfg.preload_size:
            return {}
        states, actions, rewards, next_states, dones = self.memory.sample()
        actions, rewards, dones = actions.view(-1, 1).type(torch.long), rewards.view(-1, 1), \
            dones.view(-1, 1)

        with torch.no_grad():
            a_argmax = self.net(next_states).argmax(dim=-1, keepdim=True)
            q_target = (rewards + self.cfg.gamma * (1 - dones) *
                        self.target_net(next_states).gather(-1, a_argmax)).squeeze(-1)

        q_current = self.net(states).gather(-1, actions).squeeze(-1)
        loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return {
            'loss': loss.item(),
            'q_target': q_target.mean().item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }


if __name__ == '__main__':
    cfg = Config()
    env = make_env(cfg)
    agent = DQN(cfg)
    train(env, agent, cfg)
    cfg.render_mode = 'human'
    env = make_env(cfg)
    test(env, agent, cfg)
