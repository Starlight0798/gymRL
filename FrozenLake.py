import gym
import numpy as np
import math
import torch

class Config:
    def __init__(self):
        self.env_name = 'FrozenLake-v1'
        self.algo_name = 'Q-Learning'
        self.train_eps = 400
        self.test_eps = 20
        self.max_steps = 100
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.lr = 0.1
        self.gamma = 0.9
        self.seed = 1
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

class Agent:
    def __init__(self, n_states, n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q = np.zeros((n_states, n_actions))

    def decide(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                          math.exp(-1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q[state, :])
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def update(self, state, action, reward, next_state, done):
        predict = self.Q[state, action]
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] += self.lr * (target - predict)


def env_agent_config(cfg):
    env = gym.make(cfg.env_name, render_mode = "rgb_array", map_name="4x4")
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = Agent(n_states, n_actions, cfg)
    return env, agent

def get_reward(state, next_state):
    if next_state in (5, 7, 11, 12):
        reward = -10.0
    elif next_state == 15:
        reward = 100.0
    elif state == next_state:
        reward = -5.0
    else:
        reward = -1.0
    return reward

def train(env, agent, cfg):
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []
    for i in range(cfg.train_eps):
        observation, _ = env.reset(seed = cfg.seed)
        ep_reward = 0.0
        while True:
            action = agent.decide(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            reward = get_reward(observation, next_observation)
            done = terminated or truncated
            agent.update(observation, action, reward, next_observation, done)
            observation = next_observation
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        print(f"回合：{i + 1}/{cfg.train_eps}，奖励：{ep_reward:.1f}")
    print('完成训练!')
    return rewards

if __name__ == '__main__':
    cfg = Config()
    env, agent = env_agent_config(cfg)
    rewards = train(env, agent, cfg)
    env.close()
