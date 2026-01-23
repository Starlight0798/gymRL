"""
NoisyNet DQN with Dueling Architecture for CartPole-v1

NoisyNet replaces epsilon-greedy exploration with parametric noise in network weights,
enabling learned and state-dependent exploration.

References:
- "Noisy Networks for Exploration" (Fortunato et al., 2017)
- "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)

Features:
- NoisyLinear layers with factorized Gaussian noise
- Dueling network architecture (V + A streams)
- Double DQN action selection
- No epsilon-greedy needed - exploration via noise
- Standard experience replay buffer
"""

import gymnasium as gym
import numpy as np
import random
import signal
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, List


class Config:
    def __init__(self):
        self.env_name = "CartPole-v1"
        self.seed = None
        self.max_episodes = 500
        self.max_steps = 10000
        self.batch_size = 64
        self.gamma = 0.99
        self.lr = 0.001
        self.target_update_freq = 500
        self.memory_capacity = 10000
        self.hidden_dim = 64
        self.sigma_init = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class NoisyLinear(nn.Module):
    """
    Linear layer with learnable parametric noise for exploration.
    Uses factorized Gaussian noise for efficiency.
    """

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

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_i = self._scale_noise(self.in_features)
        epsilon_j = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisyDuelingQNetwork(nn.Module):
    """
    Dueling Q-Network with NoisyLinear layers for exploration.
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        sigma_init: float = 0.5,
    ):
        super().__init__()

        self.fc1 = NoisyLinear(state_dim, hidden_dim, sigma_init)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)

        self.value_stream = NoisyLinear(hidden_dim, 1, sigma_init)
        self.advantage_stream = NoisyLinear(hidden_dim, action_dim, sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ReplayBuffer:
    """Standard experience replay buffer with uniform sampling."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
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


class NoisyDQNTrainer:
    """Trainer for NoisyNet DQN with Dueling architecture."""

    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.policy_net = NoisyDuelingQNetwork(
            state_dim, action_dim, config.hidden_dim, config.sigma_init
        ).to(config.device)
        self.target_net = NoisyDuelingQNetwork(
            state_dim, action_dim, config.hidden_dim, config.sigma_init
        ).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.memory_capacity)

        self.learn_step = 0
        self.episode_rewards = deque(maxlen=100)

        print(f"Device: {config.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if deterministic:
            self.policy_net.eval()

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)
        q_values = self.policy_net(state_tensor)

        if deterministic:
            self.policy_net.train()

        return q_values.argmax(dim=1).item()

    def update(self) -> dict:
        if len(self.memory) < self.cfg.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.cfg.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.cfg.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.cfg.device).view(
            -1, 1
        )
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.cfg.device
        ).view(-1, 1)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.cfg.device
        )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.cfg.device).view(
            -1, 1
        )

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + self.cfg.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            "loss": loss.item(),
            "q_mean": q_values.mean().item(),
        }

    def train(self):
        print("Starting training...")

        for episode in range(self.cfg.max_episodes):
            state, _ = self.env.reset(seed=self.cfg.seed)
            episode_reward = 0.0
            episode_loss = 0.0
            steps = 0

            for step in range(self.cfg.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.memory.push(state, action, reward, next_state, done)
                metrics = self.update()

                if metrics:
                    episode_loss += metrics.get("loss", 0)

                state = next_state
                episode_reward += reward
                steps += 1

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards)
            avg_loss = episode_loss / steps if steps > 0 else 0

            print(
                f"Episode {episode + 1}/{self.cfg.max_episodes} | "
                f"Reward: {episode_reward:.0f} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"Steps: {steps} | "
                f"Loss: {avg_loss:.4f}"
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
    trainer = NoisyDQNTrainer(config)

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
