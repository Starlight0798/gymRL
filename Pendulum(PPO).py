import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Beta
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils.model import MLP, PSCN, MLPRNN, ModelLoader, BaseRNNModel
from utils.buffer import ReplayBuffer_on_policy as ReplayBuffer
from utils.runner import train, test, make_env, BasicConfig
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

class Config(BasicConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.env_name = 'Pendulum-v1'
        self.algo_name = 'PPO'
        self.train_eps = 500
        self.lr_start = 1e-3
        self.lr_end = 1e-5
        self.batch_size = 1024
        self.mini_batch = 16
        self.epochs = 3
        self.clip = 0.2
        self.dual_clip = 3.0
        self.val_coef = 0.5
        self.ent_coef_start = 1e-2
        self.ent_coef_end = 1e-4
        self.ent_decay = int(0.332 * self.train_eps)
        self.grad_clip = 0.5

class ActorCritic(BaseRNNModel):
    def __init__(self, cfg):
        super(ActorCritic, self).__init__(cfg.device, hidden_size=16)
        self.device = cfg.device
        self.fc_head = PSCN(cfg.n_states, 64)
        self.rnn = MLPRNN(64, 64, batch_first=True)
        self.rnn_h = torch.zeros(1, 16, device=self.device)
        self.alpha_layer = MLP([64, cfg.n_actions])
        self.beta_layer = MLP([64, cfg.n_actions])
        self.critic_fc = MLP([64, 16, 1])

    def forward(self, s):
        x = self.fc_head(s)
        out, self.rnn_h = self.rnn(x, self.rnn_h)
        alpha = F.softplus(self.alpha_layer(out)) + 1.0
        beta = F.softplus(self.beta_layer(out)) + 1.0
        value = self.critic_fc(out)
        return alpha, beta, value
    

class PPO(ModelLoader):
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = torch.jit.script(ActorCritic(cfg).to(cfg.device))
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr_start, eps=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.train_eps // 4, eta_min=cfg.lr_end)
        self.memory = ReplayBuffer(cfg)
        self.learn_step = 0
        self.ent_coef = cfg.ent_coef_start
        self.scaler = GradScaler()
        super().__init__(save_path=f'./checkpoints/{cfg.algo_name}_{cfg.env_name}.pth')

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        alpha, beta, _ = self.net(state)
        dist = Beta(alpha, beta)
        action = dist.sample().squeeze().cpu().numpy()
        return self.fix_action(action)

    @torch.no_grad()
    def evaluate(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        alpha, beta, _ = self.net(state)
        dist = Beta(alpha, beta)
        action = dist.mean.squeeze().cpu().numpy()
        return self.fix_action(action)
    
    def fix_action(self, action):
        if action.ndim == 0:
            action = np.array([action])
        action = [2 * (a - 0.5) * self.cfg.action_bound for a in action]
        return action

    def update(self):
        if self.memory.size < self.cfg.batch_size:
            return {}
        states, actions, rewards, next_states, dones = self.memory.sample()
        rewards, dones = rewards.view(-1, 1), dones.view(-1, 1)
        if actions.ndim == 1:
            actions = actions.view(-1, 1)

        with autocast():
            self.net.reset_hidden()
            _alpha, _beta, _ = self.net(states)
            old_probs = Beta(_alpha, _beta).log_prob(actions).sum(dim=1, keepdim=True).detach()

            with torch.no_grad():
                self.net.reset_hidden()
                v = self.net(states)[-1]
                self.net.reset_hidden()
                v_ = self.net(next_states)[-1]
                td_error = rewards + self.cfg.gamma * v_ * (1 - dones) - v
                td_error = td_error.cpu().detach().numpy()
                adv, gae = [], 0.0
                for delta in td_error[::-1]:
                    gae = self.cfg.gamma * self.cfg.lamda * gae + delta
                    adv.append(gae)
                adv.reverse()
                adv = torch.tensor(np.array(adv), device=self.cfg.device, dtype=torch.float32).view(-1, 1)
                v_target = adv + v

        losses = np.zeros(5)

        for _ in range(self.cfg.epochs):
            for indices in BatchSampler(SubsetRandomSampler(range(self.memory.size)), self.cfg.mini_batch,
                                        drop_last=False):
                with autocast():
                    self.net.reset_hidden()
                    alpha, beta, value = self.net(states[indices])
                    actor_prob = Beta(alpha, beta)
                    log_probs = actor_prob.log_prob(actions[indices]).sum(dim=1, keepdim=True)
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
                    entropy_loss = -torch.mean(actor_prob.entropy())
                    loss = clip_loss + self.cfg.val_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                losses[0] += loss.item()
                losses[1] += clip_loss.item()
                losses[2] += value_loss.item()
                losses[3] += entropy_loss.item()

        self.scheduler.step()
        self.memory.clear()
        self.learn_step += 1
        self.ent_coef = self.cfg.ent_coef_end + (self.cfg.ent_coef_start - self.cfg.ent_coef_end) * \
                        np.exp(-1.0 * self.learn_step / self.cfg.ent_decay)

        return {
            'total_loss': losses[0] / self.cfg.epochs,
            'clip_loss': losses[1] / self.cfg.epochs,
            'value_loss': losses[2] / self.cfg.epochs,
            'entropy_loss': losses[3] / self.cfg.epochs / (self.cfg.batch_size // cfg.mini_batch),
            'advantage': adv.mean().item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'ent_coef': self.ent_coef
        }

if __name__ == '__main__':
    cfg = Config()
    env = make_env(cfg)
    agent = PPO(cfg)
    train(env, agent, cfg)
    cfg.render_mode = 'human'
    env = make_env(cfg)
    test(env, agent, cfg)
