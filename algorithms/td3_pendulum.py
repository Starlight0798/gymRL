"""
TD3 (Twin Delayed DDPG) for Pendulum-v1

An improved version of DDPG that addresses function approximation error through
twin critics, delayed policy updates, and target policy smoothing.
Reference: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)

Features:
- Twin critics to reduce overestimation bias
- Delayed policy updates (update actor less frequently than critic)
- Target policy smoothing (add noise to target actions)
- Clipped double Q-learning
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
        self.lr_actor = 1e-3
        self.lr_critic = 1e-3
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.exploration_noise = 0.1
        self.policy_freq = 2
        self.memory_capacity = 100000
        self.hidden_dim = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, action_bound: float
    ):
        super().__init__()
        self.action_bound = action_bound
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_bound


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

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        return self.fc3(q1)


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


class TD3Trainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.action_bound = float(self.env.action_space.high[0])

        self.actor = Actor(
            state_dim, action_dim, config.hidden_dim, self.action_bound
        ).to(config.device)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.lr_critic
        )

        self.memory = ReplayBuffer(config.memory_capacity)
        self.episode_rewards = deque(maxlen=100)
        self.total_updates = 0

        print(f"Device: {config.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Action bound: [-{self.action_bound}, {self.action_bound}]")

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
        action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        if not deterministic:
            noise = np.random.normal(
                0, self.cfg.exploration_noise * self.action_bound, size=action.shape
            )
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return action

    def update(self) -> Tuple[float, float]:
        if len(self.memory) < self.cfg.batch_size:
            return 0.0, 0.0

        self.total_updates += 1

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
            noise = (torch.randn_like(actions) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.action_bound, self.action_bound
            )
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + self.cfg.gamma * (1 - dones) * torch.min(
                target_q1, target_q2
            )

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = 0.0
        if self.total_updates % self.cfg.policy_freq == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)

            actor_loss = actor_loss.item()

        return actor_loss, critic_loss.item()

    def train(self):
        print("Starting training...")

        for episode in range(self.cfg.max_episodes):
            state, _ = self.env.reset(seed=self.cfg.seed)
            episode_reward = 0.0
            actor_loss_sum, critic_loss_sum = 0.0, 0.0
            steps = 0

            for step in range(self.cfg.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.memory.push(state, action, reward, next_state, done)
                a_loss, c_loss = self.update()
                actor_loss_sum += a_loss
                critic_loss_sum += c_loss

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
    trainer = TD3Trainer(config)

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
