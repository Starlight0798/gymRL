import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from utils.model import *
from utils.buffer import ReplayBuffer_off_policy as ReplayBuffer
from utils.runner import *
from torch.cuda.amp import autocast, GradScaler
import flappy_bird_gymnasium


class Config(BasicConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.env_name = 'FlappyBird-v0'
        self.render_mode = 'rgb_array'
        self.algo_name = 'NDQN'
        self.train_eps = 1000
        self.gamma = 0.9
        self.lr = 1e-4
        self.batch_size = 256
        self.memory_capacity = 40960
        self.target_update = 400
        self.grad_clip = 0.5
        self.load_model = False


class DQNnet(nn.Module):
    def __init__(self, cfg):
        super(DQNnet, self).__init__()
        self.head = MLP([cfg.n_states, 512, 512, 512], linear=NoisyLinear, last_act=True)
        self.fc_a = MLP([512, 128, cfg.n_actions], linear=NoisyLinear)
        self.fc_v = MLP([512, 128, 1], linear=NoisyLinear)
        
    def forward(self, obs):
        out = self.head(obs)
        V = self.fc_v(out)
        A = self.fc_a(out)
        logits = V + (A - A.mean(dim=-1, keepdim=True))
        return logits

    def reset_noise(self):
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
                

class DQN(ModelLoader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.memory = ReplayBuffer(cfg)
        self.net = DQNnet(cfg).to(cfg.device)
        self.target_net = DQNnet(cfg).to(cfg.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.scaler = GradScaler()
        self.cfg = cfg
        self.learn_step = 0
        for param in self.target_net.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float32)
        action = self.net(state).argmax(dim=-1).item()
        return action


    @torch.no_grad()
    def evaluate(self, state):
        return self.choose_action(state)


    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        actions, rewards, dones = actions.view(-1, 1).type(torch.long), rewards.view(-1, 1), \
            dones.view(-1, 1)

        with torch.no_grad():
            with autocast():
                a_argmax = self.net(next_states).argmax(dim=-1, keepdim=True)
                q_target = (rewards + self.cfg.gamma * (1 - dones) *
                            self.target_net(next_states).gather(-1, a_argmax)).squeeze(-1)

        with autocast():
            q_current = self.net(states).gather(-1, actions).squeeze(-1)
            loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.learn_step += 1
        
        if self.learn_step % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            
        self.net.reset_noise()
        self.target_net.reset_noise()

        return {
            'loss': loss.item(),
            'q_target': q_target.mean().item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }


if __name__ == '__main__':
    BenchMark.train(DQN, Config)
