import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from utils.model import *
from utils.buffer import ReplayBuffer_on_policy as ReplayBuffer
from utils.runner import *
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import flappy_bird_gymnasium

class Config(BasicConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.env_name = 'FlappyBird-v0'
        self.render_mode = 'rgb_array'
        self.algo_name = 'PPO'
        self.train_eps = 10000
        self.batch_size = 4
        self.epochs = 10
        self.clip = 0.2
        self.gamma = 0.99
        self.dual_clip = 3.0
        self.val_coef = 0.5
        self.lr_start = 1e-3
        self.lr_end = 1e-5
        self.ent_coef = 1e-2
        self.grad_clip = 0.5
        self.load_model = True


class ActorCritic(BaseRNNModel):
    def __init__(self, cfg):
        super(ActorCritic, self).__init__(device=cfg.device, hidden_size=128)
        self.fc_head = PSCN(cfg.n_states, 512)
        self.rnn = MLPRNN(512, 512, batch_first=True)
        self.actor_fc = MLP([512, 128, cfg.n_actions])
        self.critic_fc = MLP([512, 64, 1])

    def forward(self, s):
        x = self.fc_head(s)
        out, self.rnn_h = self.rnn(x, self.rnn_h)
        prob = F.softmax(self.actor_fc(out), dim=-1)
        value = self.critic_fc(out)
        return prob, value


class PPO(ModelLoader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.net = torch.jit.script(ActorCritic(cfg).to(cfg.device))
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr_start, eps=1e-5, amsgrad=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.train_eps // 4, eta_min=cfg.lr_end)
        self.memory = [ReplayBuffer(cfg) for _ in range(cfg.batch_size)]
        self.learn_step = 0
        self.scaler = GradScaler()

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
        losses = np.zeros(5)
        for _ in range(self.cfg.epochs):
            for index in np.random.permutation(self.cfg.batch_size):
                states, actions, old_probs, adv, v_target = self.memory[index].sample()
                with autocast():
                    self.net.reset_hidden()
                    actor_prob, value = self.net(states)
                    log_probs = torch.log(actor_prob.gather(1, actions))
                    ratio = torch.exp(log_probs - old_probs)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv

                    min_surr = torch.min(surr1, surr2)
                    clip_loss = -torch.mean(torch.where(
                        adv < 0,
                        torch.max(min_surr, self.cfg.dual_clip * adv),
                        min_surr
                    ))
                    value_loss = F.mse_loss(v_target, value)
                    entropy_loss = -torch.mean(-torch.sum(actor_prob * torch.log(actor_prob), dim=1))
                    loss = clip_loss + self.cfg.val_coef * value_loss + self.cfg.ent_coef * entropy_loss

                    losses[0] += loss.item()
                    losses[1] += clip_loss.item()
                    losses[2] += value_loss.item()
                    losses[3] += entropy_loss.item()

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        self.scheduler.step()
        for i in range(self.cfg.batch_size):
            self.memory[i].clear()
        self.learn_step += 1

        return {
            'total_loss': losses[0] / self.cfg.epochs / self.cfg.batch_size,
            'clip_loss': losses[1] / self.cfg.epochs / self.cfg.batch_size,
            'value_loss': losses[2] / self.cfg.epochs / self.cfg.batch_size,
            'entropy_loss': losses[3] / self.cfg.epochs / self.cfg.batch_size,
            'advantage': adv.mean().item(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

if __name__ == '__main__':
    cfg = Config()
    env = make_env(cfg)
    agent = PPO(cfg)
    train(env, agent, cfg)
    
    cfg = Config()
    cfg.render_mode = 'human'
    env = make_env(cfg)
    agent = PPO(cfg)
    test(env, agent, cfg)