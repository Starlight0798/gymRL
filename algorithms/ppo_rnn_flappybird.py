"""
PPO with RNN (GRU) for FlappyBird-v0

Proximal Policy Optimization with recurrent neural network for temporal reasoning.
Uses GRU-based actor-critic architecture with dual-clip PPO for stable learning
in partially observable environments.

References:
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- "Mastering Complex Control in MOBA Games with Deep RL" (Ye et al., 2020) - Dual Clip

Features:
- GRU-based recurrent actor-critic
- PSCN feature extraction
- MLPRNN hybrid (3:1 MLP to RNN ratio)
- Dual-clip PPO objective
- Episode-based buffer for sequence learning
- GAE advantage estimation

Requirements:
- pip install flappy_bird_gymnasium
"""

import gymnasium as gym
import numpy as np
import signal
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from typing import Tuple, List, Optional

try:
    import flappy_bird_gymnasium
except ImportError:
    print("Please install flappy_bird_gymnasium: pip install flappy_bird_gymnasium")
    sys.exit(1)


class Config:
    """Hyperparameters for PPO+RNN on FlappyBird."""

    def __init__(self):
        self.env_name = "FlappyBird-v0"
        self.seed = None
        self.max_episodes = 10000
        self.max_steps = 20000
        self.batch_size = 4  # Number of parallel episodes
        self.epochs = 10
        self.clip = 0.2
        self.dual_clip = 3.0
        self.gamma = 0.995
        self.lamda = 0.95
        self.val_coef = 0.5
        self.ent_coef = 1e-2
        self.lr = 1e-3
        self.grad_clip = 0.5
        self.eval_freq = 10
        self.save_freq = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path = "./checkpoints/PPO_RNN_FlappyBird.pth"


def initialize_weights(
    layer: nn.Module, init_type: str = "kaiming", nonlinearity: str = "leaky_relu"
) -> nn.Module:
    """Initialize layer weights."""
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
    """Multi-layer perceptron with configurable activation."""

    def __init__(
        self,
        dim_list: List[int],
        activation: nn.Module = None,
        last_act: bool = False,
    ):
        super().__init__()
        if activation is None:
            activation = nn.PReLU()

        layers = []
        for i in range(len(dim_list) - 1):
            layer = initialize_weights(nn.Linear(dim_list[i], dim_list[i + 1]))
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
    A hierarchical feature extraction network.
    """

    def __init__(self, input_dim: int, output_dim: int, depth: int = 4):
        super().__init__()
        min_dim = 2 ** (depth - 1)
        assert depth >= 1, "depth must be at least 1"
        assert output_dim >= min_dim
        assert output_dim % min_dim == 0

        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        in_dim, out_dim = input_dim, output_dim

        for i in range(depth):
            self.layers.append(MLP([in_dim, out_dim], last_act=True))
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


class MLPRNN(nn.Module):
    """
    Hybrid MLP-RNN layer with 3:1 ratio.
    Combines feedforward processing with temporal memory.
    """

    def __init__(self, input_dim: int, output_dim: int, batch_first: bool = True):
        super().__init__()
        assert output_dim % 4 == 0, "output_dim must be divisible by 4"
        self.rnn_size = output_dim // 4
        self.rnn_linear = MLP([input_dim, 3 * self.rnn_size])
        self.rnn = nn.GRU(input_dim, self.rnn_size, batch_first=batch_first)

    def forward(
        self, x: torch.Tensor, rnn_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rnn_linear_out = self.rnn_linear(x)
        rnn_out, rnn_state = self.rnn(x, rnn_state)
        out = torch.cat([rnn_linear_out, rnn_out], dim=-1)
        return out, rnn_state


class ActorCritic(nn.Module):
    """
    RNN-based Actor-Critic network with PSCN feature extraction.
    Uses JIT-compatible hidden state management.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.device = None  # Will be set when moved to device
        self.hidden_size = hidden_size
        self.rnn_h: Optional[torch.Tensor] = None

        self.fc_head = PSCN(state_dim, 512)
        self.rnn = MLPRNN(512, 512, batch_first=True)
        self.actor_fc = MLP([512, 128, action_dim])
        self.critic_fc = MLP([512, 128, 32, 1])

    def reset_hidden(self, device: Optional[str] = None):
        if device is None:
            device = next(self.parameters()).device
        self.rnn_h = torch.zeros(1, self.hidden_size, device=device, dtype=torch.float)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rnn_h is None:
            self.reset_hidden(s.device)

        x = self.fc_head(s)
        out, self.rnn_h = self.rnn(x, self.rnn_h)
        prob = F.softmax(self.actor_fc(out), dim=-1)
        value = self.critic_fc(out)
        return prob, value


class EpisodeBuffer:
    """Buffer for storing single episode transitions for on-policy learning."""

    def __init__(self, gamma: float, lamda: float, device: str):
        self.gamma = gamma
        self.lamda = lamda
        self.device = device
        self.buffer = []
        self.samples = None

    def store(self, transition):
        assert self.samples is None, (
            "Need to clear buffer before storing new transitions"
        )
        self.buffer.append(transition)

    def clear(self):
        self.buffer = []
        self.samples = None

    def size(self) -> int:
        return len(self.buffer)

    def compute_advantage(
        self, rewards, dones, dw, values, next_values
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            td_error = rewards + self.gamma * next_values * (1 - dw) - values
            td_error = td_error.cpu().detach().numpy()
            dones_np = dones.cpu().detach().numpy()
            adv, gae = [], 0.0
            for delta, d in zip(td_error[::-1], dones_np[::-1]):
                gae = self.gamma * self.lamda * gae * (1 - d) + delta
                adv.append(gae)
            adv.reverse()
            adv = torch.tensor(
                np.array(adv), device=self.device, dtype=torch.float32
            ).view(-1, 1)
            v_target = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, v_target

    def sample(self):
        if self.samples is None:
            (
                states,
                actions,
                rewards,
                dones,
                dw,
                log_probs,
                values,
                next_values,
            ) = map(
                lambda x: torch.tensor(
                    np.array(x), dtype=torch.float32, device=self.device
                ),
                zip(*self.buffer),
            )

            actions = actions.view(-1, 1).long()
            rewards = rewards.view(-1, 1)
            dones = dones.view(-1, 1)
            dw = dw.view(-1, 1)
            log_probs = log_probs.view(-1, 1)
            values = values.view(-1, 1)
            next_values = next_values.view(-1, 1)

            adv, v_target = self.compute_advantage(
                rewards, dones, dw, values, next_values
            )
            self.samples = states, actions, log_probs, adv, v_target

        return self.samples


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


class PPORNNTrainer:
    """Trainer for PPO with RNN on FlappyBird."""

    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.net = ActorCritic(state_dim, action_dim, hidden_size=128).to(config.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)

        # Episode buffers for batch_size episodes
        self.memory = [
            EpisodeBuffer(config.gamma, config.lamda, config.device)
            for _ in range(config.batch_size)
        ]

        self.state_norm = Normalization(shape=self.env.observation_space.shape)
        self.reward_scaler = RewardScaling(shape=1, gamma=config.gamma)

        self.learn_step = 0
        self.episode_rewards = deque(maxlen=100)

        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

        print(f"Device: {config.device}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")

    @torch.no_grad()
    def choose_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_tensor = torch.tensor(
            state, dtype=torch.float, device=self.cfg.device
        ).unsqueeze(0)
        prob, value = self.net(state_tensor)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def evaluate_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(
            state, dtype=torch.float, device=self.cfg.device
        ).unsqueeze(0)
        prob, _ = self.net(state_tensor)
        return prob.argmax().item()

    def update(self) -> dict:
        losses = np.zeros(5)

        for _ in range(self.cfg.epochs):
            for index in np.random.permutation(self.cfg.batch_size):
                states, actions, old_probs, adv, v_target = self.memory[index].sample()
                self.net.reset_hidden(self.cfg.device)

                actor_prob, value = self.net(states)
                dist = Categorical(actor_prob)
                log_probs = dist.log_prob(actions.squeeze(-1))

                ratio = torch.exp(log_probs.view(-1, 1) - old_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv

                # Dual-clip PPO
                min_surr = torch.min(surr1, surr2)
                clip_loss = -torch.mean(
                    torch.where(
                        adv < 0, torch.max(min_surr, self.cfg.dual_clip * adv), min_surr
                    )
                )

                value_loss = F.mse_loss(v_target, value)
                entropy_loss = -dist.entropy().mean()
                loss = (
                    clip_loss
                    + self.cfg.val_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                losses[0] += loss.item()
                losses[1] += clip_loss.item()
                losses[2] += value_loss.item()
                losses[3] += entropy_loss.item()
                losses[4] += adv.mean().item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

        for i in range(self.cfg.batch_size):
            self.memory[i].clear()
        self.learn_step += 1

        return {
            "total_loss": losses[0] / self.cfg.epochs / self.cfg.batch_size,
            "clip_loss": losses[1] / self.cfg.epochs / self.cfg.batch_size,
            "value_loss": losses[2] / self.cfg.epochs / self.cfg.batch_size,
            "entropy_loss": losses[3] / self.cfg.epochs / self.cfg.batch_size,
            "advantage": losses[4] / self.cfg.epochs / self.cfg.batch_size,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def save_model(self):
        state = {
            "net_state_dict": self.net.state_dict(),
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
            self.net.reset_hidden(self.cfg.device)

            # Get initial action
            action, log_prob, value = self.choose_action(state)

            episode_reward = 0.0
            steps = 0
            buffer_idx = episode % self.cfg.batch_size

            for step in range(self.cfg.max_steps):
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                next_state = self.state_norm(next_state)
                scaled_reward = self.reward_scaler(reward)

                # Get next action
                next_action, next_log_prob, next_value = self.choose_action(next_state)

                # Store transition
                transition = (
                    state,
                    action,
                    scaled_reward,
                    done,
                    terminated,
                    log_prob,
                    value,
                    next_value,
                )
                self.memory[buffer_idx].store(transition)

                state = next_state
                action, log_prob, value = next_action, next_log_prob, next_value
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

            # Update after batch_size episodes
            if (episode + 1) % self.cfg.batch_size == 0 and episode > 0:
                metrics = self.update()
                print(
                    f"  Update - Loss: {metrics['total_loss']:.4f}, "
                    f"Value: {metrics['value_loss']:.4f}, "
                    f"Entropy: {-metrics['entropy_loss']:.4f}"
                )

            if (episode + 1) % self.cfg.save_freq == 0:
                self.save_model()

        print("Training completed!")
        self.save_model()
        self.env.close()

    def eval(self, num_episodes: int = 10):
        print(f"\nEvaluating for {num_episodes} episodes...")
        rewards = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            state = self.state_norm(state, update=False)
            self.net.reset_hidden(self.cfg.device)

            episode_reward = 0.0
            done = False

            while not done:
                action = self.evaluate_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.state_norm(next_state, update=False)
                done = terminated or truncated
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)
            print(f"  Episode {ep + 1}: Reward = {episode_reward:.0f}")

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
        self.net.reset_hidden(self.cfg.device)

        done = False
        total_reward = 0.0

        while not done:
            action = self.evaluate_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = self.state_norm(next_state, update=False)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Visual Test Reward: {total_reward:.0f}")
        env.close()


if __name__ == "__main__":
    config = Config()
    trainer = PPORNNTrainer(config)

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
