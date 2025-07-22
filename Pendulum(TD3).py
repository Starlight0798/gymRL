import gymnasium as gym
import random
import numpy as np
import torch
from copy import deepcopy
from torch import nn, optim
from torch.nn import functional as F


class Config:
    def __init__(self):
        self.env_name = 'Pendulum-v1'
        self.algo_name = 'TD3'
        self.render_mode = 'rgb_array'
        self.train_eps = 500
        self.test_eps = 10
        self.max_steps = 200
        self.batch_size = 128
        self.memory_capacity = 10000
        self.lr_a = 2e-3
        self.lr_c = 5e-3
        self.gamma = 0.9
        self.tau = 0.005
        self.policy_freq = 2
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.seed = random.randint(0, 100)
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.n_states = None
        self.n_actions = None
        self.action_bound = None
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
        self.action_bound = torch.tensor(cfg.action_bound, dtype=torch.float32, device=cfg.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.action_bound
        return action


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc2 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc3 = nn.Linear(cfg.critic_hidden_dim, 1)

        self.fc4 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc5 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc6 = nn.Linear(cfg.critic_hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)

        q1 = F.relu(self.fc1(cat))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(cat))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    @torch.jit.export
    def Q1(self, x, a):
        cat = torch.cat([x, a], dim=1)
        q1 = F.relu(self.fc1(cat))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


class TD3:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = ReplayBuffer(cfg)
        self.total_up = 0

        self.actor = Actor(cfg).to(cfg.device)
        self.actor = torch.jit.script(self.actor)
        self.actor_target = deepcopy(self.actor)
        self.actor_target = torch.jit.script(self.actor_target)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_a)

        self.critic = Critic(cfg).to(cfg.device)
        self.critic = torch.jit.script(self.critic)
        self.critic_target = deepcopy(self.critic)
        self.critic_target = torch.jit.script(self.critic_target)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_c)
    

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().numpy()
        return action

    def update(self):
        if self.memory.size < self.cfg.batch_size:
            return 0, 0
        self.total_up += 1
        states, actions, rewards, next_states, dones = self.memory.sample()
        actions, rewards, dones = actions.view(-1, 1), rewards.view(-1, 1), dones.view(-1, 1)

        with torch.no_grad():
            noise = (torch.randn_like(actions, device=self.cfg.device) *
                    self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.cfg.action_bound, self.cfg.action_bound)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.cfg.gamma * torch.min(target_q1, target_q2)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = torch.tensor(0.0, device=self.cfg.device)

        if self.total_up % self.cfg.policy_freq == 0:
            for params in self.critic.parameters():
                params.requires_grad = False
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for params in self.critic.parameters():
                params.requires_grad = True
            self.update_params()

        return critic_loss.item(), actor_loss.item()

    def update_params(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)


def env_agent_config(cfg):
    env = gym.make(cfg.env_name, render_mode=cfg.render_mode).unwrapped
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    cfg.n_states = env.observation_space.shape[0]
    cfg.n_actions = env.action_space.shape[0]
    cfg.action_bound = env.action_space.high[0]
    agent = TD3(cfg)
    return env, agent


def train(env, agent, cfg):
    print('开始训练!')
    cfg.show()
    rewards, steps = [], []
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        critic_loss, actor_loss = 0.0, 0.0
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            c_loss, a_loss = agent.update()
            critic_loss += c_loss
            actor_loss += a_loss
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        steps.append(ep_step)
        print(f'回合:{i + 1}/{cfg.train_eps}  奖励:{ep_reward:.0f}  步数:{ep_step:.0f}'
              f'  Critic损失:{critic_loss/ep_step:.4f}  Actor损失:{actor_loss/ep_step:.4f}')
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
