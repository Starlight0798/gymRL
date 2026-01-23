"""
SAC (Soft Actor-Critic) for CartPole-v1 with Discrete Actions

Soft Actor-Critic adapted for discrete action spaces using categorical policy.
Reference: "Soft Actor-Critic for Discrete Action Settings" (Christodoulou, 2019)

Features:
- Twin Q-networks to reduce overestimation bias
- Automatic entropy temperature (alpha) tuning
- Soft policy updates via entropy regularization
- Categorical policy for discrete action selection
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
from copy import deepcopy
from collections import deque
from torch.distributions import Categorical
from typing import Tuple


class Config:
    def __init__(self):
        self.env_name = "CartPole-v1"
        self.seed = None
        self.max_episodes = 500
        self.max_steps = 2000
        self.batch_size = 128
        self.gamma = 0.9
        self.tau = 0.005
        self.lr_actor = 2e-4
        self.lr_critic = 1e-3
        self.lr_alpha = 1e-3
        self.memory_capacity = 10000
        self.hidden_dim = 256
        self.target_entropy = -1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


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


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACTrainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.actor = Actor(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.lr_actor)

        self.critic1 = Critic(state_dim, action_dim, config.hidden_dim).to(
            config.device
        )
        self.critic1_target = deepcopy(self.critic1)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=config.lr_critic)

        self.critic2 = Critic(state_dim, action_dim, config.hidden_dim).to(
            config.device
        )
        self.critic2_target = deepcopy(self.critic2)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=config.lr_critic)

        self.log_alpha = torch.tensor(
            np.log(0.01), requires_grad=True, device=config.device, dtype=torch.float32
        )
        self.alpha_optim = optim.Adam([self.log_alpha], lr=config.lr_alpha)

        self.memory = ReplayBuffer(config.memory_capacity)
        self.episode_rewards = deque(maxlen=100)

        print(f"Device: {config.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)
        probs = self.actor(state_tensor)

        if deterministic:
            return probs.argmax(dim=1).item()

        dist = Categorical(probs)
        return dist.sample().item()

    def soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.cfg.tau) + param.data * self.cfg.tau
            )

    def update(self) -> Tuple[float, float, float, float]:
        if len(self.memory) < self.cfg.batch_size:
            return 0.0, 0.0, 0.0, 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.cfg.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.cfg.device)
        actions = torch.tensor(
            actions, dtype=torch.long, device=self.cfg.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(1)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.cfg.device
        )
        dones = torch.tensor(
            dones, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(1)

        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)

            next_q1 = self.critic1_target(next_states)
            next_q2 = self.critic2_target(next_states)
            min_next_q = torch.sum(
                next_probs * torch.min(next_q1, next_q2), dim=1, keepdim=True
            )
            next_value = min_next_q + alpha * next_entropy
            target_q = rewards + self.cfg.gamma * (1 - dones) * next_value

        q1 = self.critic1(states).gather(1, actions)
        q2 = self.critic2(states).gather(1, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)

        q1_new = self.critic1(states)
        q2_new = self.critic2(states)
        min_q = torch.sum(probs * torch.min(q1_new, q2_new), dim=1, keepdim=True)
        actor_loss = torch.mean(-alpha * entropy - min_q)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = torch.mean(
            self.log_alpha.exp() * (entropy - self.cfg.target_entropy).detach()
        )

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

        return (
            actor_loss.item(),
            critic1_loss.item(),
            critic2_loss.item(),
            alpha_loss.item(),
        )

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

            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards)
            alpha = self.log_alpha.exp().item()

            print(
                f"Episode {episode + 1}/{self.cfg.max_episodes} | "
                f"Reward: {episode_reward:.0f} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"Steps: {steps} | "
                f"Alpha: {alpha:.4f}"
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
