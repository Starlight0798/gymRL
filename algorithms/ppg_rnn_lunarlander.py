"""
Phasic Policy Gradient (PPG) with RNN for LunarLander-v3

PPG separates policy and value function training into distinct phases,
using an auxiliary value head with clone loss to prevent catastrophic forgetting.
Combined with RNN for temporal reasoning in partially observable settings.

References:
- "Phasic Policy Gradient" (Cobbe et al., 2020)
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

Features:
- Dual-phase training: policy phase + auxiliary phase
- Auxiliary value head with clone loss
- GRU-based recurrent actor-critic
- PSCN feature extraction with MLPRNN
- Dual-clip PPO objective
- GAE advantage estimation
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


class Config:
    def __init__(self):
        self.env_name = "LunarLander-v3"
        self.seed = None
        self.max_episodes = 5000
        self.max_steps = 20000
        self.batch_size = 4
        self.epochs = 10
        self.aux_epochs = 6
        self.clip = 0.2
        self.dual_clip = 3.0
        self.gamma = 0.995
        self.lamda = 0.95
        self.val_coef = 0.5
        self.ent_coef = 1e-2
        self.beta_clone = 1.0
        self.lr = 1e-3
        self.grad_clip = 0.5
        self.eval_freq = 10
        self.save_freq = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path = "./checkpoints/PPG_RNN_LunarLander.pth"


def initialize_weights(layer: nn.Module) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


class MLP(nn.Module):
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
    Parallel Split Concatenate Network - hierarchical feature extractor
    that progressively splits and concatenates features across depths.
    """

    def __init__(self, input_dim: int, output_dim: int, depth: int = 4):
        super().__init__()
        min_dim = 2 ** (depth - 1)
        assert output_dim >= min_dim and output_dim % min_dim == 0

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
        return torch.cat(out_parts, dim=-1)


class MLPRNN(nn.Module):
    """Hybrid MLP-RNN with 3:1 ratio - combines fast feedforward with temporal memory."""

    def __init__(self, input_dim: int, output_dim: int, batch_first: bool = True):
        super().__init__()
        assert output_dim % 4 == 0
        self.rnn_size = output_dim // 4
        self.rnn_linear = MLP([input_dim, 3 * self.rnn_size])
        self.rnn = nn.GRU(input_dim, self.rnn_size, batch_first=batch_first)

    def forward(
        self, x: torch.Tensor, rnn_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rnn_linear_out = self.rnn_linear(x)
        rnn_out, rnn_state = self.rnn(x, rnn_state)
        return torch.cat([rnn_linear_out, rnn_out], dim=-1), rnn_state


class ActorCriticPPG(nn.Module):
    """
    PPG Actor-Critic with auxiliary value head.
    The aux_critic helps prevent value function interference during policy updates.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_h: Optional[torch.Tensor] = None

        self.fc_head = PSCN(state_dim, 256)
        self.rnn = MLPRNN(256, 256, batch_first=True)
        self.actor_fc = MLP([256, 64, action_dim])
        self.critic_fc = MLP([256, 32, 1])
        self.aux_critic_fc = MLP([256, 32, 1])

    def reset_hidden(self, device: Optional[str] = None):
        if device is None:
            device = next(self.parameters()).device
        self.rnn_h = torch.zeros(1, self.hidden_size, device=device, dtype=torch.float)

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.rnn_h is None:
            self.reset_hidden(s.device)

        x = self.fc_head(s)
        out, self.rnn_h = self.rnn(x, self.rnn_h)
        prob = F.softmax(self.actor_fc(out), dim=-1)
        value = self.critic_fc(out)
        aux_value = self.aux_critic_fc(out)
        return prob, value, aux_value


class EpisodeBuffer:
    def __init__(self, gamma: float, lamda: float, device: str):
        self.gamma = gamma
        self.lamda = lamda
        self.device = device
        self.buffer = []
        self.samples = None

    def store(self, transition):
        assert self.samples is None
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
            states, actions, rewards, dones, dw, log_probs, values, next_values = map(
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
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        return (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        return x / (self.running_ms.std + 1e-8)

    def reset(self):
        self.R = np.zeros(self.shape)


class PPGTrainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.net = ActorCriticPPG(state_dim, action_dim, hidden_size=64).to(
            config.device
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
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
        prob, value, _ = self.net(state_tensor)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def evaluate_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(
            state, dtype=torch.float, device=self.cfg.device
        ).unsqueeze(0)
        prob, _, _ = self.net(state_tensor)
        return prob.argmax().item()

    def update(self) -> dict:
        losses = np.zeros(6)

        for _ in range(self.cfg.epochs):
            for index in np.random.permutation(self.cfg.batch_size):
                states, actions, old_probs, adv, v_target = self.memory[index].sample()
                self.net.reset_hidden(self.cfg.device)

                actor_prob, value, _ = self.net(states)
                dist = Categorical(actor_prob)
                log_probs = dist.log_prob(actions.squeeze(-1))

                ratio = torch.exp(log_probs.view(-1, 1) - old_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv

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

        for _ in range(self.cfg.aux_epochs):
            for index in np.random.permutation(self.cfg.batch_size):
                states, actions, old_probs, adv, v_target = self.memory[index].sample()
                self.net.reset_hidden(self.cfg.device)

                actor_prob, _, aux_value = self.net(states)

                aux_value_loss = F.mse_loss(v_target, aux_value)

                old_probs_detached = old_probs.detach()
                current_log_probs = (
                    Categorical(actor_prob).log_prob(actions.squeeze(-1)).view(-1, 1)
                )
                clone_loss = F.mse_loss(current_log_probs, old_probs_detached)

                joint_loss = aux_value_loss + self.cfg.beta_clone * clone_loss
                losses[5] += aux_value_loss.item()

                self.optimizer.zero_grad()
                joint_loss.backward()
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
            "aux_value_loss": losses[5] / self.cfg.aux_epochs / self.cfg.batch_size,
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
                next_action, next_log_prob, next_value = self.choose_action(next_state)

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

            if (episode + 1) % self.cfg.batch_size == 0 and episode > 0:
                metrics = self.update()
                print(
                    f"  Update - Loss: {metrics['total_loss']:.4f}, "
                    f"Aux: {metrics['aux_value_loss']:.4f}"
                )

            if (episode + 1) % self.cfg.save_freq == 0:
                self.save_model()

            if avg_reward >= 200.0 and len(self.episode_rewards) >= 100:
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                break

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

        print(f"Evaluation: Mean = {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
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
    trainer = PPGTrainer(config)

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
