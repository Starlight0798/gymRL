import gymnasium as gym
import random
import numpy as np
import torch
from copy import deepcopy
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn import functional as F


class Config:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.algo_name = 'SAC'
        self.render_mode = 'rgb_array'
        self.train_eps = 500
        self.test_eps = 10
        self.max_steps = 2000
        self.batch_size = 128
        self.memory_capacity = 10000
        self.lr_a = 2e-4
        self.lr_c = 1e-3
        self.lr_alpha = 1e-3
        self.gamma = 0.9
        self.tau = 0.005
        self.seed = random.randint(0, 100)
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.n_states = None
        self.n_actions = None
        self.target_entropy = -1
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)


class ReplayBuffer:
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


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.actor_hidden_dim)
        self.fc2 = nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim)
        self.fc3 = nn.Linear(cfg.actor_hidden_dim, cfg.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = F.softmax(self.fc3(x), dim=1)
        return prob


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.critic_hidden_dim)
        self.fc2 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc3 = nn.Linear(cfg.critic_hidden_dim, cfg.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class SAC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = ReplayBuffer(cfg)

        self.actor = torch.jit.script(Actor(cfg).to(cfg.device))
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_a)

        self.critic1 = torch.jit.script(Critic(cfg).to(cfg.device))
        self.critic1_target = deepcopy(self.critic1)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=cfg.lr_c)

        self.critic2 = torch.jit.script(Critic(cfg).to(cfg.device))
        self.critic2_target = deepcopy(self.critic2)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=cfg.lr_c)

        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=cfg.device, dtype=torch.float32)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).view(1, -1)
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample().item()
        return action

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.cfg.tau) + param.data * self.cfg.tau)

    def calc_target_q(self, reward, next_state, done):
        next_action_prob = self.actor(next_state)
        next_log_prob = torch.log(next_action_prob + 1e-6)
        entropy = -torch.sum(next_action_prob * next_log_prob, dim=1, keepdim=True)
        next_q1 = self.critic1_target(next_state)
        next_q2 = self.critic2_target(next_state)
        min_q = torch.sum(next_action_prob * torch.min(next_q1, next_q2),
                          dim=1, keepdim=True)
        next_value = min_q + self.log_alpha.exp() * entropy
        target_q = reward + self.cfg.gamma * (1 - done) * next_value
        return target_q

    def update(self):
        if self.memory.size < self.cfg.batch_size:
            return 0, 0, 0, 0
        state, action, reward, next_state, done = self.memory.sample()
        action, reward, done = action.view(-1, 1).type(torch.long), \
            reward.view(-1, 1), done.view(-1, 1)
            
        target_q = self.calc_target_q(reward, next_state, done)
        q1 = self.critic1(state).gather(1, action)
        critic1_loss = torch.mean(F.mse_loss(q1, target_q.detach()))
        q2 = self.critic2(state).gather(1, action)
        critic2_loss = torch.mean(F.mse_loss(q2, target_q.detach()))
        
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        prob = self.actor(state)
        log_prob = torch.log(prob + 1e-8)
        entropy = -torch.sum(prob * log_prob, dim=1, keepdim=True)
        q1 = self.critic1(state)
        q2 = self.critic2(state)
        min_q = torch.sum(prob * torch.min(q1, q2), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_q)
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy - self.cfg.target_entropy).detach())
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        
        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), alpha_loss.item()


def env_agent_config(cfg):
    env = gym.make(cfg.env_name, render_mode=cfg.render_mode).unwrapped
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    cfg.n_states = int(env.observation_space.shape[0])
    cfg.n_actions = int(env.action_space.n)
    agent = SAC(cfg)
    return env, agent


def train(env, agent, cfg):
    print('开始训练!')
    cfg.show()
    rewards, steps = [], []
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        c1_loss, c2_loss, a_loss, alpha_loss = 0.0, 0.0, 0.0, 0.0
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            c1_loss_, c2_loss_, a_loss_, alpha_loss_ = agent.update()
            c1_loss, c2_loss, a_loss, alpha_loss = \
                c1_loss + c1_loss_, c2_loss + c2_loss_, a_loss + a_loss_, alpha_loss + alpha_loss_
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        steps.append(ep_step)
        print(f'回合:{i + 1}/{cfg.train_eps}  奖励:{ep_reward:.0f}  步数:{ep_step:.0f}  Critic1损失:{c1_loss/ep_step:.4f}  '
                f'Critic2损失:{c2_loss/ep_step:.4f}  Actor损失:{a_loss/ep_step:.4f}  '
                f'Alpha损失:{alpha_loss/ep_step:.4f}')
    print('完成训练!')
    env.close()
    return rewards, steps


def test(agent, cfg):
    print('开始测试!')
    rewards, steps = [], []
    env = gym.make(cfg.env_name, render_mode='human')
    for i in range(cfg.test_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f'回合:{i + 1}/{cfg.test_eps}, 奖励:{ep_reward:.3f}')
    print('结束测试!')
    env.close()
    return rewards, steps


if __name__ == '__main__':
    cfg = Config()
    env, agent = env_agent_config(cfg)
    train_rewards, train_steps = train(env, agent, cfg)
    test_rewards, test_steps = test(agent, cfg)
