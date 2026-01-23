"""
Double DQN with Prioritized Experience Replay and Dueling Architecture for CartPole-v1

Combines three DQN improvements:
- Double DQN: Decouples action selection from evaluation to reduce overestimation
- PER: Prioritizes important transitions based on TD-error magnitude
- Dueling: Separates value and advantage streams for better value estimation

References:
- "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
- "Prioritized Experience Replay" (Schaul et al., 2015)
- "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)

Features:
- Dueling network architecture (V + A streams)
- SumTree-based prioritized experience replay
- Double Q-learning for action selection
- Importance sampling weights for unbiased updates
- Epsilon-greedy exploration with exponential decay
"""

import gymnasium as gym
import numpy as np
import random
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple


class Config:
    def __init__(self):
        self.env_name = "CartPole-v1"
        self.seed = None
        self.max_episodes = 500
        self.max_steps = 10000
        self.batch_size = 64
        self.gamma = 0.9
        self.lr = 0.001
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 800
        self.target_update_freq = 4
        self.memory_capacity = 65536
        self.hidden_dim = 256
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.error_max = 1.0
        self.eps = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class DuelingQNetwork(nn.Module):
    """
    Dueling Network Architecture: separates Q into Value and Advantage streams.
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q


class SumTree:
    """Binary tree structure for efficient prioritized sampling in O(log n) time."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.data_pointer = 0

    def update(self, index: int, priority: float):
        change = priority - self.tree[index]
        self.tree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += change

    def add(self, priority: float, data):
        index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_leaf(self, v: float) -> Tuple[int, float, object]:
        parent_idx = 0
        while True:
            left_idx = parent_idx * 2 + 1
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                v -= self.tree[left_idx]
                parent_idx = right_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Experience replay buffer with priority-based sampling using SumTree."""

    def __init__(self, config: Config):
        self.cfg = config
        self.tree = SumTree(config.memory_capacity)

    def push(self, transition):
        max_priority = np.max(self.tree.tree[-self.cfg.memory_capacity :])
        max_priority = max_priority if max_priority != 0 else 1.0
        self.tree.add(max_priority, transition)

    def sample(self, batch_size: int) -> Tuple:
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size

        self.cfg.beta = min(1.0, self.cfg.beta + self.cfg.beta_increment)

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(v)
            priorities.append(priority)
            batch.append(data)
            indices.append(idx)

        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total_priority()
        is_weight = (self.tree.size * sampling_probs) ** (-self.cfg.beta)
        is_weight /= is_weight.max()

        return zip(*batch), np.array(indices), is_weight

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        errors = errors + self.cfg.eps
        clipped_errors = np.minimum(errors, self.cfg.error_max)
        priorities = np.power(clipped_errors, self.cfg.alpha)
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.size


class DDQNPERDuelTrainer:
    """Trainer combining Double DQN, PER, and Dueling architecture."""

    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.policy_net = DuelingQNetwork(state_dim, action_dim, config.hidden_dim).to(
            config.device
        )
        self.target_net = DuelingQNetwork(state_dim, action_dim, config.hidden_dim).to(
            config.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.memory = PrioritizedReplayBuffer(config)

        self.epsilon = config.epsilon_start
        self.sample_count = 0
        self.episode_rewards = deque(maxlen=100)

        print(f"Device: {config.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")

    def get_epsilon(self) -> float:
        self.sample_count += 1
        self.epsilon = self.cfg.epsilon_end + (
            self.cfg.epsilon_start - self.cfg.epsilon_end
        ) * np.exp(-1.0 * self.sample_count / self.cfg.epsilon_decay)
        return self.epsilon

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.get_epsilon():
            return self.env.action_space.sample()

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)
        q_values = self.policy_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def update(self) -> float:
        if len(self.memory) < self.cfg.batch_size:
            return 0.0

        (states, actions, rewards, next_states, dones), indices, is_weight = (
            self.memory.sample(self.cfg.batch_size)
        )

        states = torch.tensor(
            np.array(states), dtype=torch.float32, device=self.cfg.device
        )
        actions = torch.tensor(
            np.array(actions), dtype=torch.long, device=self.cfg.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            np.array(rewards), dtype=torch.float32, device=self.cfg.device
        )
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.cfg.device
        )
        dones = torch.tensor(
            np.array(dones), dtype=torch.float32, device=self.cfg.device
        )
        is_weight = torch.tensor(is_weight, dtype=torch.float32, device=self.cfg.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )
            target_q_values = rewards + self.cfg.gamma * next_q_values * (1 - dones)

        td_errors = q_values - target_q_values
        loss = (td_errors.pow(2) * is_weight).mean()

        priorities = td_errors.abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

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

                self.memory.push((state, action, reward, next_state, done))
                self.update()

                state = next_state
                episode_reward += reward
                steps += 1

                if done:
                    break

            if (episode + 1) % self.cfg.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards)

            print(
                f"Episode {episode + 1}/{self.cfg.max_episodes} | "
                f"Reward: {episode_reward:.0f} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"Steps: {steps} | "
                f"Epsilon: {self.epsilon:.3f}"
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
    trainer = DDQNPERDuelTrainer(config)

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
