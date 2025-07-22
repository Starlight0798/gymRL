import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils.model import *
from utils.buffer import ReplayBuffer_on_policy as ReplayBuffer
from utils.runner import *

class Config(BasicConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.env_name = 'LunarLander-v3'
        self.render_mode = 'rgb_array'
        self.algo_name = 'PPO'
        self.train_eps = 2000
        self.batch_size = 1024
        self.mini_batch = 64
        self.epochs = 10
        self.clip = 0.2
        self.gamma = 0.99
        self.dual_clip = 3.0
        self.val_coef = 0.5
        self.lr = 1e-4
        self.ent_coef = 1e-2
        self.grad_clip = 0.5
        self.load_model = False


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super(ActorCritic, self).__init__()
        self.fc_head = PSCN(cfg.n_states, 256, depth=4)
        self.actor_fc = MLP([256, 64, cfg.n_actions])
        self.critic_fc = MLP([256, 64, 1])

    def forward(self, s):
        x = self.fc_head(s)
        prob = F.softmax(self.actor_fc(x), dim=-1)
        value = self.critic_fc(x)
        return prob, value


class PPO(ModelLoader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.net = torch.jit.script(ActorCritic(cfg).to(cfg.device))
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.memory = ReplayBuffer(cfg)
        self.learn_step = 0

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        prob, value = self.net(state)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def evaluate(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        prob, _ = self.net(state)
        m = Categorical(prob)
        action = m.probs.argmax().item()
        return action
    
    def update(self):
        states, actions, old_probs, adv, v_target = self.memory.sample()
        losses = np.zeros(5)

        for _ in range(self.cfg.epochs):
            for indices in BatchSampler(SubsetRandomSampler(range(self.memory.size())), self.cfg.mini_batch, drop_last=False):
                actor_prob, value = self.net(states[indices])
                dist = Categorical(actor_prob)
                log_probs = dist.log_prob(actions[indices])
                ratio = torch.exp(log_probs - old_probs[indices])
                surr1 = ratio * adv[indices]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv[indices]

                min_surr = torch.min(surr1, surr2)
                clip_loss = -torch.mean(torch.where(
                    adv[indices] < 0,
                    torch.max(min_surr, self.cfg.dual_clip * adv[indices]),
                    min_surr
                ))
                value_loss = F.mse_loss(v_target[indices], value)
                entropy_loss = -dist.entropy().mean()
                loss = clip_loss + self.cfg.val_coef * value_loss + self.cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

                losses[0] += loss.item()
                losses[1] += clip_loss.item()
                losses[2] += value_loss.item()
                losses[3] += entropy_loss.item()
                
        self.memory.clear()
        self.learn_step += 1

        return {
            'total_loss': losses[0] / self.cfg.epochs,
            'clip_loss': losses[1] / self.cfg.epochs,
            'value_loss': losses[2] / self.cfg.epochs,
            'entropy_loss': losses[3] / self.cfg.epochs / (self.cfg.batch_size // self.cfg.mini_batch),
            'advantage': adv.mean().item(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

if __name__ == '__main__':
    BenchMark.train(PPO, Config)
