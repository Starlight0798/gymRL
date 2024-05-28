import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils.model import MLP, PSCN, MLPRNN, ModelLoader, ConvBlock
from utils.buffer import ReplayBuffer_on_policy as ReplayBuffer
from utils.runner import train, test, make_env, BasicConfig
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

class Config(BasicConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.env_name = 'MsPacmanDeterministic-v4'
        self.render_mode = 'rgb_array'
        self.max_steps = 1000
        self.algo_name = 'PPO'
        self.train_eps = 15000
        self.batch_size = 512
        self.mini_batch = 16
        self.epochs = 3
        self.clip = 0.2
        self.dual_clip = 3.0
        self.val_coef = 0.5
        self.lr_start = 1e-3
        self.lr_end = 1e-5
        self.ent_coef_start = 1e-2
        self.ent_coef_end = 1e-5
        self.ent_decay = int(0.332 * self.train_eps)
        self.grad_clip = 0.5
        self.load_model = True
        self.use_atari = True
        self.save_freq = 50

class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super(ActorCritic, self).__init__()
        self.device = cfg.device
        self.conv_layer = ConvBlock(
            channels=[(3, 16), (16, 32), (32, 16)],
            output_dim=256,
            input_shape=(3, 84, 84),
        )
        self.fc_head = PSCN(256, 256)
        self.rnn = MLPRNN(256, 256, batch_first=True)
        self.rnn_h = torch.zeros(1, 64, device=self.device)
        self.actor_fc = MLP([256, 32, cfg.n_actions])
        self.critic_fc = MLP([256, 32, 1])

    def forward(self, s):
        feature = self.conv_layer(s)
        x = self.fc_head(feature)
        out, self.rnn_h = self.rnn(x, self.rnn_h)
        prob = F.softmax(self.actor_fc(out), dim=1)
        value = self.critic_fc(out)
        return prob, value

    @torch.jit.export
    def reset_hidden(self):
        self.rnn_h = torch.zeros(1, 64, device=self.device)

class PPO(ModelLoader):
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = torch.jit.script(ActorCritic(cfg).to(cfg.device))
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr_start, eps=1e-5, amsgrad=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.train_eps // 4, eta_min=cfg.lr_end)
        self.memory = ReplayBuffer(cfg)
        self.learn_step = 0
        self.ent_coef = cfg.ent_coef_start
        self.scaler = GradScaler()
        super().__init__(save_path=f'./checkpoints/{cfg.algo_name}_{cfg.env_name}.pth')

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        prob, _ = self.net(state)
        m = Categorical(prob)
        action = m.sample().item()
        return action

    @torch.no_grad()
    def evaluate(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        prob, _ = self.net(state)
        m = Categorical(prob)
        action = m.probs.argmax().item()
        return action

    def update(self):
        if self.memory.size < self.cfg.batch_size:
            return {}
        states, actions, rewards, next_states, dones = self.memory.sample()
        actions, rewards, dones = actions.view(-1, 1).type(torch.long), \
            rewards.view(-1, 1), dones.view(-1, 1)

        with autocast():
            self.net.reset_hidden()
            old_probs = torch.log(self.net(states)[0].gather(1, actions)).detach()

            with torch.no_grad():
                self.net.reset_hidden()
                _, v = self.net(states)
                self.net.reset_hidden()
                _, v_ = self.net(next_states)
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
                    actor_prob, value = self.net(states[indices])
                    log_probs = torch.log(actor_prob.gather(1, actions[indices]))
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
                    entropy_loss = -torch.mean(-torch.sum(actor_prob * torch.log(actor_prob), dim=1))
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
            'lr': self.optimizer.param_groups[0]['lr']
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
