"""
PPO (Proximal Policy Optimization) for LunarLander-v3

A clean implementation of PPO with clipped surrogate objective and GAE.
Reference: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

Features:
- Clipped surrogate objective for stable policy updates
- Dual-clip for preventing excessively large policy updates when advantage < 0
- Generalized Advantage Estimation (GAE) for variance reduction
- Entropy bonus for exploration
- Learning rate annealing
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
from torch.distributions import Categorical
from collections import deque
from typing import List, Tuple, Optional


class Config:
    def __init__(self):
        self.env_name = "LunarLander-v3"
        self.seed = None

        self.max_train_steps = 1_000_000
        self.update_freq = 2048
        self.num_epochs = 10
        self.batch_size = 64

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.dual_clip = 3.0
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        self.lr = 3e-4
        self.anneal_lr = True

        self.hidden_dim = 256

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def layer_init(layer: nn.Module, std: float = np.sqrt(2)) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.squeeze().item()

    def get_value(self, state: torch.Tensor) -> float:
        _, value = self.forward(state)
        return value.squeeze().item()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


class PPOTrainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.model = ActorCritic(state_dim, action_dim, config.hidden_dim).to(
            config.device
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)

        self.buffer = RolloutBuffer()
        self.step_count = 0
        self.episode_rewards = deque(maxlen=100)

        print(f"Device: {config.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.buffer.rewards)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values + [next_value])

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t] + self.cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            )
            advantages[t] = last_gae = (
                delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * last_gae
            )

        returns = advantages + values[:-1]
        return advantages, returns

    def collect_rollout(self):
        self.buffer.clear()
        state, _ = self.env.reset(seed=self.cfg.seed)
        episode_reward = 0.0

        for _ in range(self.cfg.update_freq):
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.cfg.device
            ).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = self.model.get_action(state_tensor)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(state, action, log_prob, value, reward, done)

            state = next_state
            episode_reward += reward
            self.step_count += 1

            if done:
                self.episode_rewards.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0.0

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)
        with torch.no_grad():
            next_value = self.model.get_value(state_tensor)

        return next_value

    def update(self, next_value: float) -> dict:
        advantages, returns = self.compute_gae(next_value)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(
            np.array(self.buffer.states), dtype=torch.float32, device=self.cfg.device
        )
        actions = torch.tensor(
            self.buffer.actions, dtype=torch.long, device=self.cfg.device
        )
        old_log_probs = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.cfg.device
        )
        advantages_t = torch.tensor(
            advantages, dtype=torch.float32, device=self.cfg.device
        )
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.cfg.device)

        batch_size = len(self.buffer)
        indices = np.arange(batch_size)

        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fracs = []
        approx_kls = []

        for epoch in range(self.cfg.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                new_log_probs, values, entropy = self.model.evaluate_actions(
                    mb_states, mb_actions
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                    * mb_advantages
                )

                min_surr = torch.min(surr1, surr2)
                policy_loss = -torch.mean(
                    torch.where(
                        mb_advantages < 0,
                        torch.max(min_surr, self.cfg.dual_clip * mb_advantages),
                        min_surr,
                    )
                )

                value_loss = self.cfg.value_coef * torch.mean(
                    (values - mb_returns).pow(2)
                )

                entropy_loss = -self.cfg.entropy_coef * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.mean().item())

                with torch.no_grad():
                    clip_frac = torch.mean(
                        (
                            (ratio < 1 - self.cfg.clip_eps)
                            | (ratio > 1 + self.cfg.clip_eps)
                        ).float()
                    ).item()
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    clip_fracs.append(clip_frac)
                    approx_kls.append(approx_kl)

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "clip_frac": np.mean(clip_fracs),
            "approx_kl": np.mean(approx_kls),
        }

    def train(self):
        print("Starting training...")
        update_count = 0

        while self.step_count < self.cfg.max_train_steps:
            if self.cfg.anneal_lr:
                frac = 1.0 - self.step_count / self.cfg.max_train_steps
                lr = self.cfg.lr * frac
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            next_value = self.collect_rollout()

            metrics = self.update(next_value)
            update_count += 1

            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards)
                print(
                    f"Step: {self.step_count:,} | "
                    f"Updates: {update_count} | "
                    f"Avg Reward: {avg_reward:.1f} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"Clip: {metrics['clip_frac']:.2%}"
                )

                if avg_reward >= 200.0 and len(self.episode_rewards) >= 100:
                    print(f"\nEnvironment solved at step {self.step_count:,}!")
                    break

        print("Training completed!")
        self.env.close()

    def eval(self, num_episodes: int = 10):
        print(f"\nEvaluating for {num_episodes} episodes...")
        self.model.eval()
        rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.cfg.device
                ).unsqueeze(0)

                with torch.no_grad():
                    action, _, _ = self.model.get_action(
                        state_tensor, deterministic=True
                    )

                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.1f}")

        print(
            f"Evaluation Results: Mean = {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}"
        )
        self.model.train()
        return rewards

    def test(self):
        self.eval(num_episodes=5)

        print("\nStarting visual test...")
        env = gym.make(self.cfg.env_name, render_mode="human")
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        self.model.eval()

        while not done:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.cfg.device
            ).unsqueeze(0)

            with torch.no_grad():
                action, _, _ = self.model.get_action(state_tensor, deterministic=True)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Visual Test Reward: {total_reward:.1f}")
        env.close()
        self.model.train()


if __name__ == "__main__":
    config = Config()
    trainer = PPOTrainer(config)

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
