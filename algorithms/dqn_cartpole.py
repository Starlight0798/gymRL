"""
DQN (Deep Q-Network) for CartPole-v1

A classic implementation of DQN with experience replay and target network.
Reference: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

Features:
- Experience replay buffer for stable learning
- Target network with periodic updates
- Epsilon-greedy exploration with exponential decay
- Gradient clipping for stability
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
        self.max_steps = 500
        self.batch_size = 64
        self.gamma = 0.99
        self.lr = 1e-3
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 800
        self.target_update_freq = 4
        self.memory_capacity = 100000
        self.hidden_dim = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def layer_init(layer: nn.Module, std: float = np.sqrt(2)) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNTrainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.policy_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(
            config.device
        )
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(
            config.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.memory_capacity)

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

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.cfg.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.cfg.device)
        actions = torch.tensor(
            actions, dtype=torch.long, device=self.cfg.device
        ).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.cfg.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.cfg.device
        )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.cfg.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.cfg.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

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

                self.memory.push(state, action, reward, next_state, done)
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
            f"Evaluation Results: Mean = {np.mean(rewards):.1f} ± {np.std(rewards):.1f}"
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
    trainer = DQNTrainer(config)

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
