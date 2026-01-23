"""
Rainbow DQN for CartPole-v1

Rainbow DQN combining multiple improvements: Double DQN, Dueling architecture,
Prioritized Experience Replay, N-step returns, and NoisyNet.
Reference: "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2018)

Features:
- NoisyNet for parameter-space exploration
- Dueling network architecture (V + A streams)
- Prioritized experience replay with importance sampling
- N-step returns for better credit assignment
- Double DQN for reduced overestimation
- Soft target network updates
"""

import gymnasium as gym
import numpy as np
import math
import random
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from collections import deque
from typing import Tuple, Dict


class Config:
    def __init__(self):
        self.env_name = "CartPole-v1"
        self.seed = None
        self.max_episodes = 500
        self.max_steps = 500
        self.batch_size = 256
        self.gamma = 0.9
        self.tau = 0.005
        self.lr = 1e-3
        self.memory_capacity = 20000
        self.hidden_dim = 256
        self.n_steps = 5
        self.alpha = 0.6
        self.beta_init = 0.4
        self.grad_clip = 10.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingNoisyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = NoisyLinear(hidden_dim, action_dim)
        self.value = NoisyLinear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_capacity = 2 * capacity - 1
        self.tree = np.zeros(self.tree_capacity)

    def update(self, data_index: int, priority: float):
        tree_index = data_index + self.capacity - 1
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_index(self, v: float) -> Tuple[int, float]:
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            if left_idx >= self.tree_capacity:
                tree_index = parent_idx
                break
            if v <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                v -= self.tree[left_idx]
                parent_idx = right_idx
        data_index = tree_index - self.capacity + 1
        return data_index, self.tree[tree_index]

    @property
    def priority_sum(self) -> float:
        return self.tree[0]

    @property
    def priority_max(self) -> float:
        return self.tree[self.capacity - 1 :].max()


class PrioritizedNStepBuffer:
    def __init__(self, config: Config, state_dim: int):
        self.device = config.device
        self.capacity = config.memory_capacity
        self.batch_size = config.batch_size
        self.n_steps = config.n_steps
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.beta = config.beta_init
        self.beta_init = config.beta_init

        self.sum_tree = SumTree(self.capacity)
        self.n_step_deque = deque(maxlen=self.n_steps)

        self.buffer = {
            "state": np.zeros((self.capacity, state_dim)),
            "action": np.zeros((self.capacity, 1)),
            "reward": np.zeros(self.capacity),
            "next_state": np.zeros((self.capacity, state_dim)),
            "terminal": np.zeros(self.capacity),
        }
        self.current_size = 0
        self.count = 0

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        done: bool,
    ):
        transition = (state, action, reward, next_state, terminal, done)
        self.n_step_deque.append(transition)

        if len(self.n_step_deque) == self.n_steps:
            state, action, n_step_reward, next_state, terminal = (
                self._get_n_step_transition()
            )
            self.buffer["state"][self.count] = state
            self.buffer["action"][self.count] = action
            self.buffer["reward"][self.count] = n_step_reward
            self.buffer["next_state"][self.count] = next_state
            self.buffer["terminal"][self.count] = terminal

            priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
            self.sum_tree.update(self.count, priority)

            self.count = (self.count + 1) % self.capacity
            self.current_size = min(self.current_size + 1, self.capacity)

    def _get_n_step_transition(self) -> Tuple:
        state, action = self.n_step_deque[0][:2]
        next_state, terminal = self.n_step_deque[-1][3:5]
        n_step_reward = 0.0

        for i in reversed(range(self.n_steps)):
            r, s_, ter, d = self.n_step_deque[i][2:]
            n_step_reward = r + self.gamma * (1 - d) * n_step_reward
            if d:
                next_state, terminal = s_, ter

        return state, action, n_step_reward, next_state, terminal

    def sample(
        self, total_steps: int, max_train_steps: int
    ) -> Tuple[Dict, np.ndarray, torch.Tensor]:
        batch_index = np.zeros(self.batch_size, dtype=np.int64)
        is_weight = torch.zeros(
            self.batch_size, dtype=torch.float32, device=self.device
        )

        segment = self.sum_tree.priority_sum / self.batch_size
        self.beta = self.beta_init + (1 - self.beta_init) * (
            total_steps / max_train_steps
        )

        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            index, priority = self.sum_tree.get_index(v)
            batch_index[i] = index
            prob = priority / self.sum_tree.priority_sum
            is_weight[i] = (self.current_size * prob) ** (-self.beta)

        is_weight /= is_weight.max()

        batch = {}
        for key in self.buffer.keys():
            if key == "action":
                batch[key] = torch.tensor(
                    self.buffer[key][batch_index], dtype=torch.long, device=self.device
                )
            else:
                batch[key] = torch.tensor(
                    self.buffer[key][batch_index],
                    dtype=torch.float32,
                    device=self.device,
                )

        return batch, batch_index, is_weight

    def update_priorities(self, batch_index: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(index, priority)

    def __len__(self) -> int:
        return self.current_size


class RainbowDQNTrainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.max_steps_per_episode = self.env.spec.max_episode_steps

        self.max_train_steps = self.max_steps_per_episode * config.max_episodes

        self.policy_net = DuelingNoisyNetwork(
            self.state_dim, self.action_dim, config.hidden_dim
        ).to(config.device)
        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.memory = PrioritizedNStepBuffer(config, self.state_dim)

        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)

        print(f"Device: {config.device}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic:
            self.total_steps += 1

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)

        if deterministic:
            self.policy_net.eval()

        q_values = self.policy_net(state_tensor)

        if deterministic:
            self.policy_net.train()

        return q_values.argmax(dim=-1).item()

    def update(self) -> float:
        if len(self.memory) < self.cfg.batch_size:
            return 0.0

        batch, batch_index, is_weight = self.memory.sample(
            self.total_steps, self.max_train_steps
        )

        with torch.no_grad():
            next_actions = self.policy_net(batch["next_state"]).argmax(
                dim=-1, keepdim=True
            )
            next_q = (
                self.target_net(batch["next_state"])
                .gather(-1, next_actions)
                .squeeze(-1)
            )
            target_q = (
                batch["reward"]
                + (self.cfg.gamma**self.cfg.n_steps) * (1 - batch["terminal"]) * next_q
            )

        current_q = (
            self.policy_net(batch["state"]).gather(-1, batch["action"]).squeeze(-1)
        )
        td_error = current_q - target_q

        loss = (td_error.pow(2) * is_weight).mean()

        self.memory.update_priorities(batch_index, td_error.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        for param, target_param in zip(
            self.policy_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data
            )

        lr_now = (
            0.9 * self.cfg.lr * (1 - self.total_steps / self.max_train_steps)
            + 0.1 * self.cfg.lr
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now

        return loss.item()

    def train(self):
        print("Starting training...")

        for episode in range(self.cfg.max_episodes):
            state, _ = self.env.reset(seed=self.cfg.seed)
            episode_reward = 0.0
            steps = 0

            for step in range(self.cfg.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                terminal = done and step != self.max_steps_per_episode - 1

                self.memory.store_transition(
                    state, action, reward, next_state, terminal, done
                )
                self.update()

                state = next_state
                episode_reward += reward
                steps += 1

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards)

            print(
                f"Episode {episode + 1}/{self.cfg.max_episodes} | "
                f"Reward: {episode_reward:.0f} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"Steps: {steps}"
            )

            if avg_reward >= 495.0 and len(self.episode_rewards) >= 100:
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                break

        print("Training completed!")
        self.env.close()

    def eval(self, num_episodes: int = 10):
        print(f"\nEvaluating for {num_episodes} episodes...")
        rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.0f}")

        print(
            f"Evaluation Results: Mean = {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}"
        )
        return rewards

    def test(self):
        self.eval(num_episodes=5)

        print("\nStarting visual test...")
        env = gym.make(self.cfg.env_name, render_mode="human")
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = self.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Visual Test Reward: {total_reward:.0f}")
        env.close()


if __name__ == "__main__":
    config = Config()
    trainer = RainbowDQNTrainer(config)

    def signal_handler(signum, frame):
        print("\n\nTraining interrupted. Starting test...")
        trainer.test()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    trainer.test()
