import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils.model import *
from utils.buffer import ReplayBuffer_on_policy as ReplayBuffer
from utils.runner import *
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
        self.batch_size = 4096
        self.mini_batch = 64
        self.epochs = 3
        self.clip = 0.2
        self.gamma = 0.99
        self.dual_clip = 3.0
        self.val_coef = 0.5
        self.lr_start = 1e-3
        self.lr_end = 1e-5
        self.ent_coef = 2e-2
        self.grad_clip = 0.5
        self.use_atari = True
        self.save_freq = 50
        self.load_model = True


class ActorCritic(BaseRNNModel):
    def __init__(self, cfg):
        super(ActorCritic, self).__init__(device=cfg.device, hidden_size=128)
        self.conv_layer = ConvBlock(
            channels=[(3, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512)],
            output_dim=512,
            input_shape=(3, 84, 84),
            use_depthwise=False
        )
        self.fc_head = PSCN(512, 512)
        self.rnn = MLPRNN(512, 512, batch_first=True)
        self.actor_fc = MLP([512, 128, cfg.n_actions])
        self.critic_fc = MLP([512, 64, 1])

    def forward(self, s):
        feature = self.conv_layer(s)
        x = self.fc_head(feature)
        out, self.rnn_h = self.rnn(x, self.rnn_h)
        prob = F.softmax(self.actor_fc(out), dim=1)
        value = self.critic_fc(out)
        return prob, value


class PPO(ModelLoader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.net = torch.jit.script(ActorCritic(cfg).to(cfg.device))
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr_start, eps=1e-5, amsgrad=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.train_eps // 4, eta_min=cfg.lr_end)
        self.memory = ReplayBuffer(cfg)
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
        if self.memory.size < self.cfg.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones, old_probs, values, next_values = self.memory.sample()
        with autocast():
            with torch.no_grad():
                td_error = rewards + self.cfg.gamma * next_values * (1 - dones) - values
                td_error = td_error.cpu().detach().numpy()
                adv, gae = [], 0.0
                for delta in td_error[::-1]:
                    gae = self.cfg.gamma * self.cfg.lamda * gae + delta
                    adv.append(gae)
                adv.reverse()
                adv = torch.tensor(np.array(adv), device=self.cfg.device, dtype=torch.float32).view(-1, 1)
                v_target = adv + values

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
                    loss = clip_loss + self.cfg.val_coef * value_loss + self.cfg.ent_coef * entropy_loss

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

        return {
            'total_loss': losses[0] / self.cfg.epochs,
            'clip_loss': losses[1] / self.cfg.epochs,
            'value_loss': losses[2] / self.cfg.epochs,
            'entropy_loss': losses[3] / self.cfg.epochs / (self.cfg.batch_size // cfg.mini_batch),
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