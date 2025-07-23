import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import gymnasium as gym
from collections import deque
import warnings
import signal
import sys
from typing import List, Type, Optional

warnings.filterwarnings('ignore', category=UserWarning)

# 配置类
class Config:
    def __init__(self):
        # 环境参数
        self.env_name = "LunarLander-v3"
        self.seed = None
        
        # 训练参数
        self.max_train_steps = 5e6      # 最大训练步数
        self.update_freq = 4096         # 每次更新前收集的经验数
        self.num_epochs = 4             # 每次更新时的epoch数
        self.seq_len = 8                # RNN处理的序列长度
        self.batch_size = 128           # 批次大小(序列的数量)
        self.gamma = 0.995              # 折扣因子
        self.lam_actor = 0.95           # GAE参数 - actor
        self.lam_critic = 0.97          # GAE参数 - critic
        self.clip_eps_min = 0.2         # PPO-CLIP-MIN参数 
        self.clip_eps_max = 0.28        # PPO-CLIP-MAX参数
        self.clip_cov_ratio = 0.2       # PPO-COV-RATIO参数
        self.clip_cov_min = 1.0         # PPO-COV-MIN参数
        self.clip_cov_max = 5.0         # PPO-COV-MAX参数
        self.dual_clip = 3.0            # 双重裁剪
        self.entropy_coef = 0.01        # 熵奖励系数
        self.lr = 3e-4                  # 学习率
        self.max_grad_norm = 0.5        # 梯度裁剪阈值
        self.anneal = False             # 是否退火
        self.device = 'cpu'
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = PSCN(
            input_dim=state_dim, 
            output_dim=256, 
            depth=4,
        )
        
        self.rnn = URNN(
            input_size=256, 
            hidden_size=256, 
            layer=nn.LSTM,
        )   
        
        self.actor = MLP([256, 256, action_dim], last_std=0.001)
        self.critic = MLP([256, 256, 1], last_std=1.0)
        self.rnd = RND(
            input_dim=state_dim,
            embed_dim=256,
        )
            
    def forward(self, x, hidden_state):
        """
        可以处理单步或序列数据。
        x: (batch, state_dim) 或 (batch, seq_len, state_dim)
        hidden_state: (batch, hidden_dim)
        """
        # 检查输入是单步还是序列
        is_sequence = x.dim() == 3
        if not is_sequence:
            # 如果是单步，增加一个时间维度，使其成为长度为1的序列
            x = x.unsqueeze(1)

        predict, target = self.rnd(x)
        x = self.shared(x)
        rnn_out, new_hidden_state = self.rnn(x, hidden_state)
        logits = self.actor(rnn_out)
        value = self.critic(rnn_out)

        if not is_sequence:
            # 如果输入是单步，则移除时间维度以保持输出格式一致
            logits = logits.squeeze(1)
            value = value.squeeze(1)
            predict = predict.squeeze(1)
            target = target.squeeze(1)
        
        return logits, value, new_hidden_state, predict, target
    
    def get_action(self, x, hidden_state, deterministic=False):
        """采样动作并返回相关数据和新的隐藏状态"""
        logits, value, new_hidden_state, predict, target = self.forward(x, hidden_state)
        dist = Categorical(logits=logits)
        action = dist.sample() if not deterministic else logits.argmax(dim=-1)
        log_prob = dist.log_prob(action)
        # 使用 .squeeze(-1) 更安全，避免当 batch_size=1 时维度被完全压缩
        return action.cpu().item(), log_prob.cpu().item(), value.squeeze(-1).cpu().item(), new_hidden_state, predict.cpu().numpy(), target.cpu().numpy()

    def get_value(self, x, hidden_state):
        """获取状态价值"""
        _, value, _, _, _ = self.forward(x, hidden_state)
        return value.squeeze(-1).cpu()

def layer_init(layer, std: float):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer

def make_fc_layer(
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    std: float = np.sqrt(2),
):
    layer = nn.Linear(in_features, out_features, bias=use_bias)
    return layer_init(layer, std=std)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class MLP(nn.Module):
    def __init__(
        self,
        dim_list: List[int],
        activation: Type[nn.Module] = nn.SiLU,
        last_act: bool = False,
        last_std: Optional[float] = None,
        norm: Optional[Type[nn.Module]] = RMSNorm,
    ):
        super(MLP, self).__init__()
        assert dim_list, "Dim list can't be empty!"
        layers = []
        for i in range(len(dim_list) - 1):
            if last_std and i == len(dim_list) - 2:
                layer = make_fc_layer(dim_list[i], dim_list[i + 1], std=last_std)
            else:
                layer = make_fc_layer(dim_list[i], dim_list[i + 1])
            layers.append(layer)
            if i < len(dim_list) - 2:
                layers.append(activation(inplace=True))
                if norm:
                    layers.append(norm(dim_list[i + 1]))

        if last_act:
            layers.append(activation(inplace=True))
            if norm:
                layers.append(norm(dim_list[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class PSCN(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        depth: int, 
    ):
        super(PSCN, self).__init__()
        min_dim = 2 ** (depth - 1)
        assert depth >= 1, "depth must be at least 1"
        assert output_dim >= min_dim, f"output_dim must be >= {min_dim} for depth {depth}"
        assert output_dim % min_dim == 0, f"output_dim must be divisible by {min_dim} for depth {depth}"
        
        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        in_dim, out_dim = input_dim, output_dim
        
        for _ in range(depth):
            self.layers.append(MLP([in_dim, out_dim], last_act=True))
            in_dim = out_dim // 2
            out_dim //= 2 

    def forward(self, x: torch.Tensor):
        out_parts = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                split_size = self.output_dim // (2 ** (i + 1))
                part, x = torch.split(x, [split_size, split_size], dim=-1)
                out_parts.append(part)
            else:
                out_parts.append(x)

        out = torch.cat(out_parts, dim=-1)
        return out

class URNN(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        layer: Type[nn.Module],
        *args,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = layer(input_size=input_size, hidden_size=hidden_size, batch_first=True, *args, **kwargs)
        if isinstance(self.rnn, nn.LSTM):
            self.chunk_size = 2
        elif isinstance(self.rnn, nn.GRU):
            self.chunk_size = 1
        elif isinstance(self.rnn, sLSTM):
            self.chunk_size = 4
        else:
            raise ValueError("Unsupported RNN layer type.")
        
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor]):
        """
        x: (batch_size, seq_len, input_size)
        hidden_state: (batch_size, hidden_size * chunk_size)
        """
        batch_size = x.size(0)
        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_size * self.chunk_size, device=x.device)
        else:
            # (batch, hidden_dim) -> (1, batch, hidden_dim)
            hidden_state = hidden_state.unsqueeze(0)
        
        if self.chunk_size > 1:
            h_in = torch.chunk(hidden_state, self.chunk_size, dim=-1)
            rnn_out, h_out = self.rnn(x, tuple(h_in))
            new_hidden_state = torch.cat(h_out, dim=-1)
        else:
            h_in = hidden_state
            rnn_out, h_out = self.rnn(x, h_in)
            new_hidden_state = h_out
        
        # rnn_out: (batch, seq_len, hidden_size)
        # new_hidden_state: (1, batch, hidden_dim) squeeze 成 (batch, hidden_dim)
        new_hidden_state = new_hidden_state.squeeze(0)
            
        return rnn_out, new_hidden_state
    
class RND(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(RND, self).__init__()
        min_dim = 2 ** 4
        assert np.log2(embed_dim).is_integer(), "embed_dim must be a power of 2"
        assert embed_dim >= min_dim, f"embed_dim must be at least {min_dim}"
        assert embed_dim % min_dim == 0, f"embed_dim must be divisible by {min_dim}"
        depth = np.log2(embed_dim // min_dim).astype(int)
        self.predictor, self.target = [
            PSCN(
                input_dim=input_dim, 
                output_dim=embed_dim,
                depth=depth
            ) for _ in range(2)
        ]
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        predict = self.predictor(x)
        with torch.no_grad():
            target = self.target(x)
        return predict, target

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        assert in_features % num_blocks == 0
        assert out_features % num_blocks == 0

        block_in_features = in_features // num_blocks
        block_out_features = out_features // num_blocks

        self.blocks = nn.ModuleList([
            nn.Linear(block_in_features, block_out_features) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = x.chunk(self.num_blocks, dim=-1)
        x = [block(x_i) for block, x_i in zip(self.blocks, x)]
        x = torch.cat(x, dim=-1)
        return x

class sLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=4 / 3):
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor

        assert hidden_size % num_heads == 0
        assert proj_factor > 0

        self.input_norm = RMSNorm(input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)

        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)

        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

        self.up_proj_left = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.up_proj_right = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.down_proj = nn.Linear(int(hidden_size * proj_factor), input_size)

    def forward(self, x, prev_state):
        assert x.size(-1) == self.input_size
        h_prev, c_prev, n_prev, m_prev = prev_state
        x_norm = self.input_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t

        output = h_t
        output_norm = self.group_norm(output)
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = F.gelu(output_right)
        output = output_left * output_gated
        output = self.down_proj(output)
        final_output = output + x

        return final_output, (h_t, c_t, n_t, m_t)

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=4 / 3):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.proj_factor_slstm = proj_factor
        self.layers = nn.ModuleList([sLSTMBlock(input_size, hidden_size, num_heads, proj_factor) for _ in range(num_layers)])

    def forward(self, x, state=None):
        assert x.ndim == 3
        if self.batch_first: x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        if state is not None:
            state = torch.stack(list(state)).to(x.device)
            assert state.ndim == 4
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4
            assert state_num_layers == self.num_layers
            assert state_batch_size == batch_size
            assert state_input_size == self.input_size
            state = state.transpose(0, 1)
        else:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size, device=x.device)

        output = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))
            output.append(x_t)

        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        state = tuple(state.transpose(0, 1))
        return output, state

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = [] 
        self.hidden_states = []
        self.next_value = None
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.hidden_states.clear()
        self.next_value = None

class PPOTrainer:
    def __init__(self, config):
        self.cfg = config
        # 要求 update_freq 必须是 seq_len 的整数倍
        assert self.cfg.update_freq % self.cfg.seq_len == 0, \
            f"update_freq ({self.cfg.update_freq}) must be divisible by seq_len ({self.cfg.seq_len})"
        # 批次大小现在是序列的数量，所以它必须能整除总序列数
        self.num_sequences = self.cfg.update_freq // self.cfg.seq_len
        assert self.num_sequences % self.cfg.batch_size == 0, \
            f"num_sequences ({self.num_sequences}) must be divisible by batch_size ({self.cfg.batch_size})"

        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.model = ActorCritic(state_dim, action_dim).to(self.cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.hidden_size = self.model.rnn.hidden_size * self.model.rnn.chunk_size
        print(f"Model hidden size: {self.hidden_size}")
        
        self.step_count = 0
        self.episode_rewards = deque(maxlen=10)
        self.lr = self.cfg.lr
        self.ent_coef = self.cfg.entropy_coef
        
        print(f"Using device: {self.cfg.device}")
        print(f"Sequence length: {self.cfg.seq_len}")

    def collect_experience(self):
        """收集经验数据 (逻辑基本不变)"""
        self.buffer = RolloutBuffer()
        seed = random.randint(1, 2**31 - 1) if self.cfg.seed is None else self.cfg.seed
        state, _ = self.env.reset(seed=seed)
        episode_reward = 0
        
        hidden_state = torch.zeros(1, self.hidden_size, device=self.cfg.device)
        
        for _ in range(self.cfg.update_freq):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
            
            # 存储当前步开始前的隐藏状态
            self.buffer.hidden_states.append(hidden_state.squeeze(0).cpu().numpy())

            with torch.no_grad():
                action, log_prob, value, next_hidden_state, predict, target = self.model.get_action(state_tensor, hidden_state)
                
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            rnd_reward = np.mean((predict - target) ** 2)
            episode_reward += reward
            reward += rnd_reward
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(log_prob)
            self.buffer.values.append(value)
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)
            
            state = next_state
            hidden_state = next_hidden_state 
            self.step_count += 1
            
            if done:
                self.episode_rewards.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0
                # 在 episode 结束时重置隐藏状态
                hidden_state = torch.zeros(1, self.hidden_size, device=self.cfg.device)
                
        with torch.no_grad():
            self.buffer.next_value = self.model.get_value(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device),
                hidden_state
            ).item()

    def compute_advantages(self):
        """计算GAE优势估计"""
        rewards = np.array(self.buffer.rewards)
        dones = np.array(self.buffer.dones)
        values = np.array(self.buffer.values + [self.buffer.next_value])
        
        adv_actor = np.zeros_like(rewards)
        adv_critic = np.zeros_like(rewards)
        last_gae_actor = 0
        last_gae_critic = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            adv_actor[t] = last_gae_actor = delta + self.cfg.gamma * self.cfg.lam_actor * (1 - dones[t]) * last_gae_actor
            adv_critic[t] = last_gae_critic = delta + self.cfg.gamma * self.cfg.lam_critic * (1 - dones[t]) * last_gae_critic
        returns = adv_critic + values[:-1]

        return adv_actor, returns

    def update_model(self, advantages, returns):
        """
        执行基于序列的PPO参数更新
        """
        # 1. 将所有经验数据转换为Tensor
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.cfg.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.cfg.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.cfg.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.cfg.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32).to(self.cfg.device)
        hidden_states = torch.tensor(np.array(self.buffer.hidden_states), dtype=torch.float32).to(self.cfg.device)

        # 2. 将扁平数据重塑为序列
        # (update_freq, dim) -> (num_sequences, seq_len, dim)
        states = states.view(self.num_sequences, self.cfg.seq_len, -1)
        actions = actions.view(self.num_sequences, self.cfg.seq_len)
        old_log_probs = old_log_probs.view(self.num_sequences, self.cfg.seq_len)
        advantages = advantages.view(self.num_sequences, self.cfg.seq_len)
        returns = returns.view(self.num_sequences, self.cfg.seq_len)
        old_values = old_values.view(self.num_sequences, self.cfg.seq_len)
        
        # 3. 提取每个序列的初始隐藏状态
        # 我们只需要每个序列开始时的那个隐藏状态
        initial_hidden_states = hidden_states[::self.cfg.seq_len]

        policy_losses, value_losses, entropy_losses, rnd_losses = [], [], [], []
        approx_kls, clip_fracs, covs_list = [], [], []
        
        for _ in range(self.cfg.num_epochs):
            # 4. 对序列进行混洗，而不是对单个时间步
            perm = torch.randperm(self.num_sequences, device=self.cfg.device)
            
            for start in range(0, self.num_sequences, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                idx = perm[start:end]

                # 5. 获取一个批次的序列数据
                s_batch = states[idx]
                a_batch = actions[idx]
                old_lp_batch = old_log_probs[idx]
                adv_batch = advantages[idx]
                ret_batch = returns[idx]
                hidden_batch = initial_hidden_states[idx]
                val_batch = old_values[idx]

                # 6. 模型前向传播，一次处理整个序列
                # s_batch: (batch_size, seq_len, state_dim)
                # hidden_batch: (batch_size, hidden_dim)
                logits, values, _, predict, target = self.model(s_batch, hidden_batch)
                # logits: (batch_size, seq_len, action_dim)
                # values: (batch_size, seq_len, 1)
                
                # 7. 将输出和标签展平以计算损失
                # (batch_size, seq_len, dim) -> (batch_size * seq_len, dim)
                logits_flat = logits.view(-1, logits.shape[-1])
                values_flat = values.view(-1)
                a_batch_flat = a_batch.view(-1)
                old_lp_batch_flat = old_lp_batch.view(-1)
                adv_batch_flat = adv_batch.view(-1)
                ret_batch_flat = ret_batch.view(-1)
                old_values_flat = val_batch.view(-1)

                dist = Categorical(logits=logits_flat)
                new_log_probs = dist.log_prob(a_batch_flat)
                
                ratio = (new_log_probs - old_lp_batch_flat).exp()
                covs = (new_log_probs - new_log_probs.mean()) * (adv_batch_flat - adv_batch_flat.mean())
                corr = torch.ones_like(adv_batch_flat)
                
                clip_ratio = ratio.clamp(0.0, self.cfg.dual_clip)
                surr1 = clip_ratio * adv_batch_flat
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps_min, 1 + self.cfg.clip_eps_max) * adv_batch_flat
                clip_idx = torch.where((covs > self.cfg.clip_cov_min) & (covs < self.cfg.clip_cov_max))[0]
                if len(clip_idx) > 0 and self.cfg.clip_cov_ratio > 0:
                    clip_num = max(int(len(clip_idx) * self.cfg.clip_cov_ratio), 1)
                    clip_idx = clip_idx[torch.randperm(len(clip_idx))[:min(clip_num, len(clip_idx))]]
                    corr[clip_idx] = 0.0
                clip_frac = torch.mean(((ratio < (1 - self.cfg.clip_eps_min)) | (ratio > (1 + self.cfg.clip_eps_max))).float() * corr)
                policy_loss = torch.mean(-torch.min(surr1, surr2) * corr)
                
                value_clip = old_values_flat + (values_flat - old_values_flat).clamp(-self.cfg.clip_eps_min, self.cfg.clip_eps_max)
                value_loss1 = (values_flat - ret_batch_flat).pow(2)
                value_loss2 = (value_clip - ret_batch_flat).pow(2)
                value_loss = 0.5 * torch.mean(corr * torch.max(value_loss1, value_loss2))
                
                entropy = (dist.entropy() * corr).mean()
                entropy_loss = self.ent_coef * -entropy
                
                rnd_loss = (predict - target).pow(2).mean()
                
                loss = policy_loss + value_loss + entropy_loss + rnd_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                rnd_losses.append(rnd_loss.item())
                clip_fracs.append(clip_frac.item())
                covs_list.append(covs.mean().item())
                
                with torch.no_grad():
                    approx_kl = (old_lp_batch_flat - new_log_probs).mean().item()
                    approx_kls.append(approx_kl)
        
        if self.cfg.anneal:
            self.lr = self.cfg.lr * (1 - self.step_count / self.cfg.max_train_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            self.ent_coef = self.cfg.entropy_coef * (1 - self.step_count / self.cfg.max_train_steps)
        
        print(
            f"Policy Loss: {np.mean(policy_losses):.4f} | "
            f"Value Loss: {np.mean(value_losses):.4f} | "
            f"Entropy: {np.mean(entropy_losses):.4f} | "
            f"RND Loss: {np.mean(rnd_losses):.4f} | "
            f"KL: {np.mean(approx_kls):.4f} | "
            f"Clip Frac: {np.mean(clip_fracs):.2%} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Ent Coef: {self.ent_coef:.4f} | "
            f"Cov: {np.mean(covs_list):.4f}"
        )
                
    def train(self):
        """执行训练循环"""
        while self.step_count < self.cfg.max_train_steps:
            self.collect_experience()
            advantages, returns = self.compute_advantages()
            self.update_model(advantages, returns)
            
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                print(f"Step: {self.step_count}, Avg Reward: {mean_reward:.2f}")

    def eval(self, num_episodes=10):
        """评估模型性能"""
        self.model.eval()
        total_rewards = []
        for _ in range(num_episodes):
            seed = random.randint(1, 2**31 - 1) if self.cfg.seed is None else self.cfg.seed
            state, _ = self.env.reset(seed=seed)
            episode_reward = 0
            done = False
            hidden_state = torch.zeros(1, self.hidden_size, device=self.cfg.device)
            while not done:
                with torch.no_grad():
                    action, _, _, hidden_state, _, _ = self.model.get_action(
                        torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device), 
                        hidden_state,
                        deterministic=True
                    )
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            total_rewards.append(episode_reward)
        print(f"Test Results: Mean Reward {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        self.model.train()
        
    def test(self):
        """测试模型性能, 可视化渲染"""
        self.eval(num_episodes=10)
        self.model.eval()
        seed = random.randint(1, 2**31 - 1) if self.cfg.seed is None else self.cfg.seed
        env = gym.make(self.cfg.env_name, render_mode='human')
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        hidden_state = torch.zeros(1, self.hidden_size, device=self.cfg.device)
        
        while not done:
            env.render()
            with torch.no_grad():
                action, _, _, hidden_state, _, _ = self.model.get_action(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device), 
                    hidden_state,
                    deterministic=True
                )
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Total Reward: {total_reward:.2f}")
        env.close()
        self.model.train()

if __name__ == "__main__":
    config = Config()
    ppo = PPOTrainer(config)

    def signal_handler(signum, frame):
        print("\n检测到 Ctrl+C，停止训练并开始测试...")
        ppo.test()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        ppo.train()
    except KeyboardInterrupt:
        # 捕获 KeyboardInterrupt 以便在手动停止时也能测试
        pass
    
    print("\n训练完成或被中断，开始最终测试...")
    ppo.test()

