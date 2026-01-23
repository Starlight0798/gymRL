"""
SAC (Soft Actor-Critic) for Pendulum-v1

A maximum entropy reinforcement learning algorithm for continuous action spaces
that optimizes a stochastic policy with entropy regularization.
Reference: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor" (Haarnoja et al., 2018)

Features:
- Maximum entropy framework for robust exploration
- Twin critics to reduce overestimation bias
- Automatic temperature (alpha) tuning
- Reparameterization trick for continuous actions
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
from copy import deepcopy
from typing import Tuple


class Config:
    def __init__(self):
        self.env_name = "Pendulum-v1"
        self.seed = None
        self.max_episodes = 500
        self.max_steps = 200
        self.batch_size = 128
        self.gamma = 0.99
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.lr_alpha = 3e-4
        self.tau = 0.005
        self.init_alpha = 0.2
        self.memory_capacity = 100000
        self.hidden_dim = 256
        self.log_std_min = -20
        self.log_std_max = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_bound: float,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        self.action_bound = action_bound
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_bound

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_bound * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean) * self.action_bound
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = torch.tanh(normal.rsample()) * self.action_bound
        return action


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=1)

        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2


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


class SACTrainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.action_bound = float(self.env.action_space.high[0])

        self.actor = Actor(
            state_dim,
            action_dim,
            config.hidden_dim,
            self.action_bound,
            config.log_std_min,
            config.log_std_max,
        ).to(config.device)

        self.critic = Critic(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.lr_critic
        )

        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(
            np.log(config.init_alpha), requires_grad=True, device=config.device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr_alpha)

        self.memory = ReplayBuffer(config.memory_capacity)
        self.episode_rewards = deque(maxlen=100)

        print(f"Device: {config.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Action bound: [-{self.action_bound}, {self.action_bound}]")
        print(f"Target entropy: {self.target_entropy}")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.cfg.tau * source_param.data
                + (1.0 - self.cfg.tau) * target_param.data
            )

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)
        action = (
            self.actor.get_action(state_tensor, deterministic).squeeze(0).cpu().numpy()
        )
        return action

    def update(self) -> Tuple[float, float, float]:
        if len(self.memory) < self.cfg.batch_size:
            return 0.0, 0.0, 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.cfg.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.cfg.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.cfg.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(1)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.cfg.device
        )
        dones = torch.tensor(
            dones, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.cfg.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_probs - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic_target, self.critic)

        return actor_loss.item(), critic_loss.item(), alpha_loss.item()

    def train(self):
        print("Starting training...")

        for episode in range(self.cfg.max_episodes):
            state, _ = self.env.reset(seed=self.cfg.seed)
            episode_reward = 0.0
            actor_loss_sum, critic_loss_sum, alpha_loss_sum = 0.0, 0.0, 0.0
            steps = 0

            for step in range(self.cfg.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.memory.push(state, action, reward, next_state, done)
                a_loss, c_loss, al_loss = self.update()
                actor_loss_sum += a_loss
                critic_loss_sum += c_loss
                alpha_loss_sum += al_loss

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
                f"Steps: {steps} | "
                f"Alpha: {self.alpha.item():.3f} | "
                f"Actor Loss: {actor_loss_sum / steps:.4f} | "
                f"Critic Loss: {critic_loss_sum / steps:.4f}"
            )

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
    trainer = SACTrainer(config)

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
