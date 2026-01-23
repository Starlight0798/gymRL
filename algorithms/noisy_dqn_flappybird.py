"""
NoisyNet DQN with Dueling Architecture for FlappyBird-v0

NoisyNet replaces epsilon-greedy exploration with parametric noise in network weights,
enabling learned and state-dependent exploration. Uses PSCN (Parallel Split Concatenate
Network) for efficient feature extraction.

References:
- "Noisy Networks for Exploration" (Fortunato et al., 2017)
- "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)

Features:
- NoisyLinear layers with factorized Gaussian noise
- PSCN feature extraction network
- Dueling architecture (V + A streams)
- Double DQN action selection
- Gradient clipping for stability

Requirements:
- pip install flappy_bird_gymnasium
"""

import gymnasium as gym
import numpy as np
import random
import signal
import sys
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, List, Optional

try:
    import flappy_bird_gymnasium
except ImportError:
    print("Please install flappy_bird_gymnasium: pip install flappy_bird_gymnasium")
    sys.exit(1)


class Config:
    """Hyperparameters for NoisyNet DQN on FlappyBird."""

    def __init__(self):
        self.env_name = "FlappyBird-v0"
        self.seed = None
        self.max_episodes = 1000
        self.max_steps = 20000
        self.batch_size = 256
        self.gamma = 0.9
        self.lr = 1e-4
        self.target_update_freq = 400
        self.memory_capacity = 51200
        self.sigma_init = 0.5
        self.grad_clip = 1.0
        self.eval_freq = 10
        self.save_freq = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path = "./checkpoints/NoisyDQN_FlappyBird.pth"


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
            "weight_epsilon",
            torch.FloatTensor(out_features, in_features),
            persistent=False,
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer(
            "bias_epsilon", torch.FloatTensor(out_features), persistent=False
        )

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


def initialize_weights(
    layer: nn.Module, init_type: str = "kaiming", nonlinearity: str = "leaky_relu"
) -> nn.Module:
    """Initialize layer weights using specified initialization method."""
    if isinstance(layer, nn.Linear):
        if init_type == "kaiming":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(layer.weight)
        elif init_type == "orthogonal":
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation and layer type."""

    def __init__(
        self,
        dim_list: List[int],
        activation: nn.Module = None,
        last_act: bool = False,
        linear: type = nn.Linear,
    ):
        super().__init__()
        if activation is None:
            activation = nn.PReLU()

        layers = []
        for i in range(len(dim_list) - 1):
            layer = linear(dim_list[i], dim_list[i + 1])
            if linear == nn.Linear:
                layer = initialize_weights(layer)
            layers.append(layer)
            if i < len(dim_list) - 2:
                layers.append(activation)
        if last_act:
            layers.append(activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PSCN(nn.Module):
    """
    Parallel Split Concatenate Network.
    A hierarchical feature extraction network that balances width and depth.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 4,
        linear: type = nn.Linear,
    ):
        super().__init__()
        min_dim = 2 ** (depth - 1)
        assert depth >= 1, "depth must be at least 1"
        assert output_dim >= min_dim, (
            f"output_dim must be >= {min_dim} for depth {depth}"
        )
        assert output_dim % min_dim == 0, (
            f"output_dim must be divisible by {min_dim} for depth {depth}"
        )

        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        in_dim, out_dim = input_dim, output_dim

        for i in range(depth):
            self.layers.append(MLP([in_dim, out_dim], last_act=True, linear=linear))
            in_dim = out_dim // 2
            out_dim //= 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_parts = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                split_size = int(self.output_dim // (2 ** (i + 1)))
                part, x = torch.split(x, [split_size, split_size], dim=-1)
                out_parts.append(part)
            else:
                out_parts.append(x)

        out = torch.cat(out_parts, dim=-1)
        return out


class NoisyDuelingDQN(nn.Module):
    """
    Dueling DQN with PSCN feature extractor and NoisyLinear layers.
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """

    def __init__(self, state_dim: int, action_dim: int, sigma_init: float = 0.5):
        super().__init__()
        # PSCN head with NoisyLinear
        self.head = nn.Sequential(
            PSCN(state_dim, 512, linear=NoisyLinear),
            MLP([512, 256, 256], linear=NoisyLinear, last_act=True),
        )

        # Advantage and Value streams
        self.fc_a = MLP([256, 64, action_dim], linear=NoisyLinear)
        self.fc_v = MLP([256, 64, 1], linear=NoisyLinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        V = self.fc_v(out)
        A = self.fc_a(out)
        logits = V + (A - A.mean(dim=-1, keepdim=True))
        return logits

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int):
        self.buffer = np.empty(capacity, dtype=object)
        self.pointer = 0
        self.capacity = capacity
        self.is_full = False

    def push(self, state, action, reward, next_state, done):
        self.buffer[self.pointer] = (state, action, reward, next_state, done)
        self.pointer = (self.pointer + 1) % self.capacity
        if self.pointer == 0:
            self.is_full = True

    def sample(self, batch_size: int) -> Tuple:
        size = self.capacity if self.is_full else self.pointer
        indices = np.random.choice(size, batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self) -> int:
        return self.capacity if self.is_full else self.pointer


class RunningMeanStd:
    """Dynamically calculate running mean and std."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x, dtype=np.float32)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    """State normalization using running statistics."""

    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


class RewardScaling:
    """Reward scaling using running statistics."""

    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)
        return x

    def reset(self):
        self.R = np.zeros(self.shape)


class NoisyDQNTrainer:
    """Trainer for NoisyNet DQN with Dueling architecture on FlappyBird."""

    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.net = NoisyDuelingDQN(state_dim, action_dim, config.sigma_init).to(
            config.device
        )
        self.target_net = NoisyDuelingDQN(state_dim, action_dim, config.sigma_init).to(
            config.device
        )
        self.target_net.load_state_dict(self.net.state_dict())

        # Disable gradients for target network
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.net.train()
        self.target_net.train()

        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.memory_capacity)

        self.state_norm = Normalization(shape=self.env.observation_space.shape)
        self.reward_scaler = RewardScaling(shape=1, gamma=config.gamma)

        self.learn_step = 0
        self.episode_rewards = deque(maxlen=100)

        # Create checkpoint directory
        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

        print(f"Device: {config.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if deterministic:
            self.net.eval()

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).view(1, -1)
        q_values = self.net(state_tensor)

        if deterministic:
            self.net.train()

        return q_values.argmax(dim=-1).item()

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

        # Double DQN: select action with policy net, evaluate with target net
        with torch.no_grad():
            a_argmax = self.net(next_states).argmax(dim=-1, keepdim=True)
            q_target = (
                rewards
                + self.cfg.gamma
                * (1 - dones)
                * self.target_net(next_states).gather(-1, a_argmax)
            ).squeeze(-1)

        q_current = self.net(states).gather(-1, actions).squeeze(-1)
        loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return {
            "loss": loss.item(),
            "q_target": q_target.mean().item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def save_model(self):
        state = {
            "net_state_dict": self.net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "learn_step": self.learn_step,
            "state_norm": self.state_norm,
        }
        torch.save(state, self.cfg.save_path)
        print(f"Model saved to {self.cfg.save_path}")

    def load_model(self):
        if os.path.exists(self.cfg.save_path):
            checkpoint = torch.load(self.cfg.save_path, map_location=self.cfg.device)
            self.net.load_state_dict(checkpoint["net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.learn_step = checkpoint["learn_step"]
            if "state_norm" in checkpoint:
                self.state_norm = checkpoint["state_norm"]
            print(f"Model loaded from {self.cfg.save_path}")
        else:
            print(f"No checkpoint found at {self.cfg.save_path}")

    def train(self):
        print("Starting training...")

        for episode in range(self.cfg.max_episodes):
            state, _ = self.env.reset(seed=self.cfg.seed)
            state = self.state_norm(state)
            self.reward_scaler.reset()

            episode_reward = 0.0
            episode_loss = 0.0
            steps = 0

            for step in range(self.cfg.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward

                # Normalize and scale
                next_state = self.state_norm(next_state)
                scaled_reward = self.reward_scaler(reward)

                self.memory.push(state, action, scaled_reward, next_state, done)
                metrics = self.update()

                if metrics:
                    episode_loss += metrics.get("loss", 0)

                state = next_state
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

            if (episode + 1) % self.cfg.save_freq == 0:
                self.save_model()

        print("Training completed!")
        self.save_model()
        self.env.close()

    def eval(self, num_episodes: int = 10):
        print(f"\nEvaluating for {num_episodes} episodes...")
        rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.state_norm(state, update=False)
            episode_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.state_norm(next_state, update=False)
                done = terminated or truncated
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.0f}")

        print(
            f"Evaluation Results: Mean = {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}"
        )
        return rewards

    def test(self):
        self.load_model()
        self.eval(num_episodes=5)

        print("\nStarting visual test...")
        env = gym.make(self.cfg.env_name, render_mode="human")
        state, _ = env.reset()
        state = self.state_norm(state, update=False)
        done = False
        total_reward = 0.0

        while not done:
            action = self.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = self.state_norm(next_state, update=False)
            done = terminated or truncated
            state = next_state
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
