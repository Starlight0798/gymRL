import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
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
        
        # mHC 参数
        self.use_mhc = True             # 是否使用 mHC
        self.mhc_dim = 256              # mHC 特征维度 (Increased for better capacity)
        self.mhc_rate = 2               # mHC 扩展率 (branches)
        self.mhc_layers = 2             # mHC 层数
        self.mhc_sk_it = 10             # Sinkhorn-Knopp 迭代次数 (Reduced for speed/stability)
        
        # 训练参数
        self.max_train_steps = 5e6      # 最大训练步数
        self.update_freq = 4096         # 每次更新前收集的经验数
        self.num_epochs = 4             # 每次更新时的epoch数
        self.seq_len = 8                # RNN处理的序列长度
        self.batch_size = 128           # 批次大小(序列的数量)
        self.gamma = 0.995              # 折扣因子
        self.lam_actor = 0.95           # GAE参数 - actor
        self.lam_critic = 0.95          # GAE参数 - critic
        self.clip_eps_min = 0.2         # PPO-CLIP-MIN参数 
        self.clip_eps_max = 0.28        # PPO-CLIP-MAX参数
        self.clip_cov_ratio = 0.0       # PPO-COV-RATIO参数
        self.clip_cov_min = 1.0         # PPO-COV-MIN参数
        self.clip_cov_max = 5.0         # PPO-COV-MAX参数
        self.dual_clip = 3.0            # 双重裁剪
        self.entropy_coef = 0.015       # 熵奖励系数
        self.erc_beta_low = 0.06        # ERC 下限
        self.erc_beta_high = 0.06       # ERC 上限
        self.lr = 3e-4                  # 学习率
        self.max_grad_norm = 0.5        # 梯度裁剪阈值
        self.anneal = True              # 是否退火
        self.device = 'cpu'
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config=None):
        super().__init__()
        
        shared_out_dim = 512
        
        if config and hasattr(config, 'use_mhc') and config.use_mhc:
            self.shared = MHCBackbone(
                input_dim=state_dim,
                output_dim=config.mhc_dim,
                rate=config.mhc_rate,
                num_layers=config.mhc_layers,
                max_sk_it=config.mhc_sk_it
            )
            shared_out_dim = config.mhc_dim
        else:
            self.shared = PSCN(
                input_dim=state_dim, 
                output_dim=512, 
                depth=5,
            )
            shared_out_dim = 512
        
        self.rnn = URNN(
            input_size=shared_out_dim, 
            hidden_size=512, 
            layer=nn.LSTM,
        )   
        
        self.actor = MLP([512, 512, action_dim], last_std=0.001)
        self.critic = MLP([512, 512, 1], last_std=1.0)
        self.rnd = RND(
            input_dim=state_dim,
            embed_dim=512,
        )
            
    def forward(self, x, hidden_state):
        is_sequence = x.dim() == 3
        if not is_sequence:
            x = x.unsqueeze(1)

        predict, target = self.rnd(x)
        x = self.shared(x)
        rnn_out, new_hidden_state = self.rnn(x, hidden_state)
        logits = self.actor(rnn_out)
        value = self.critic(rnn_out)

        if not is_sequence:
            logits = logits.squeeze(1)
            value = value.squeeze(1)
            predict = predict.squeeze(1)
            target = target.squeeze(1)
        
        return logits, value, new_hidden_state, predict, target
    
    def get_action(self, x, hidden_state, deterministic=False):
        logits, value, new_hidden_state, predict, target = self.forward(x, hidden_state)
        dist = Categorical(logits=logits)
        action = dist.sample() if not deterministic else logits.argmax(dim=-1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.cpu().item(), log_prob.cpu().item(), value.squeeze(-1).cpu().item(), new_hidden_state, predict.cpu().numpy(), target.cpu().numpy(), entropy.cpu().item()

    def get_value(self, x, hidden_state):
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


# --- mHC Components Start ---

def sinkhorn_knopp_batched(A, it=1000, eps=1e-8):
    """
    Batched Sinkhorn-Knopp algorithm to project matrix A onto the Birkhoff polytope (doubly stochastic matrices).
    A should be a non-negative matrix.
    """
    batch_size, n, _, = A.shape
    
    u = torch.ones(batch_size, n, device=A.device)
    v = torch.ones(batch_size, n, device=A.device)
    
    for _ in range(it):
        v_temp = v.unsqueeze(2)  # (B, n, 1)
        Av = torch.bmm(A, v_temp).squeeze(2)  # (B, n)
        u = 1.0 / (Av + eps)
        
        u_temp = u.unsqueeze(2)  # (B, n, 1)
        At_u = torch.bmm(A.transpose(1, 2), u_temp).squeeze(2)
        v = 1.0 / (At_u + eps)
        
    U = torch.diag_embed(u)  # (B, n, n)
    V = torch.diag_embed(v)  # (B, n, n)
    P = torch.bmm(torch.bmm(U, A), V)
    
    return P, U, V


class ManifoldHyperConnectionFuse(nn.Module):
    """
    mHC Fusion Layer.
    h: hyper hidden matrix (BxLxNxD)
    """
    def __init__(self, dim, rate, max_sk_it):
        super(ManifoldHyperConnectionFuse, self).__init__()

        self.n = rate
        self.dim = dim

        self.nc = self.n * self.dim
        self.n2 = self.n * self.n

        # norm flatten
        self.norm = RMSNorm(dim*rate)

        # parameters
        self.w = nn.Parameter(torch.zeros(self.nc, self.n2 + 2*self.n))
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)
        
        # Initialize beta to favor identity mapping for H_res (mixing matrix)
        # This prevents branch collapse at initialization
        beta_init = torch.zeros(self.n2 + 2*self.n)
        beta_init[:2*self.n] = 0.01 # H_pre and H_post small init
        
        # H_res: initialize close to Identity to keep branches independent initially
        res_beta = torch.zeros(self.n, self.n)
        res_beta.fill_(-2.0) # Suppress off-diagonal
        res_beta.fill_diagonal_(2.0) # Encourage diagonal
        beta_init[2*self.n:] = res_beta.flatten()
        
        self.beta = nn.Parameter(beta_init)

        # max sinkhorn knopp iterations
        self.max_sk_it = max_sk_it

    def mapping(self, h, res_norm):
        B, L, N, D = h.shape

        # 1.vectorize
        h_vec_flat = h.reshape(B, L, N*D)
        
        # RMSNorm Fused Trick
        h_vec = self.norm.weight * h_vec_flat

        # 2.projection
        H = h_vec @ self.w

        # RMSNorm Fused: compute r
        r = h_vec_flat.norm(dim=-1, keepdim=True) / math.sqrt(self.nc)
        r_ = 1.0 / (r + 1e-6)
        
        # 4. mapping
        n = N
        H_pre = r_ * H[:,:, :n] * self.alpha[0] + self.beta[:n]
        H_post = r_ * H[:,:, n:2*n] * self.alpha[1] + self.beta[n:2*n]
        H_res = r_ * H[:,:, 2*n:] * self.alpha[2] + self.beta[2*n:]

        # 5. final constrained mapping 
        H_pre = torch.sigmoid(H_pre)
        H_post = 2 * torch.sigmoid(H_post)

        # 6. sinkhorn_knopp iteration
        H_res = H_res.reshape(B, L, N, N)
        H_res_exp = H_res.exp()
        
        with torch.no_grad():
            _, U, V = res_norm(H_res_exp.reshape(B*L, N, N), self.max_sk_it)
        # recover
        P = torch.bmm(torch.bmm(U.detach(), H_res_exp.reshape(B*L, N, N)), V.detach())
        H_res = P.reshape(B, L, N, N)

        return H_pre, H_post, H_res

    def process(self, h, H_pre, H_res):
        # Weighted sum of branches
        h_pre = H_pre.unsqueeze(dim=2) @ h
        
        # Inter-branch mixing
        h_res = H_res @ h
        return h_pre, h_res

    def depth_connection(self, h_res, h_out, beta):
        # Broadcast output back to branches
        post_mapping = beta.unsqueeze(dim=-1) @ h_out
        out = post_mapping + h_res
        return out


class MHCBlock(nn.Module):
    def __init__(self, dim, rate, max_sk_it):
        super(MHCBlock, self).__init__()
        # Mixing features (similar to Attention)
        self.linear1 = nn.Linear(dim, dim)
        self.mhc1 = ManifoldHyperConnectionFuse(dim=dim, rate=rate, max_sk_it=max_sk_it)
        
        # FFN equivalent
        self.linear2 = nn.Linear(dim, dim)
        self.mhc2 = ManifoldHyperConnectionFuse(dim=dim, rate=rate, max_sk_it=max_sk_it)
        
        self.act = nn.SiLU()

    def forward(self, h):
        # h: [B, L, N, D]
        
        # Block 1
        H_pre, H_post, H_res = self.mhc1.mapping(h, sinkhorn_knopp_batched)
        h_pre, h_res = self.mhc1.process(h, H_pre, H_res)
        # h_pre: [B, L, 1, D]
        h_out = self.linear1(h_pre)
        h_out = self.act(h_out)
        h = self.mhc1.depth_connection(h_res, h_out, beta=H_post)
        
        # Block 2
        H_pre, H_post, H_res = self.mhc2.mapping(h, sinkhorn_knopp_batched)
        h_pre, h_res = self.mhc2.process(h, H_pre, H_res)
        # h_pre: [B, L, 1, D]
        h_out = self.linear2(h_pre)
        h_out = self.act(h_out)
        h = self.mhc2.depth_connection(h_res, h_out, beta=H_post)
        
        return h

class MHCBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, rate, num_layers, max_sk_it):
        super(MHCBackbone, self).__init__()
        self.rate = rate
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # MHC Layers
        self.layers = nn.ModuleList([
            MHCBlock(dim=output_dim, rate=rate, max_sk_it=max_sk_it)
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(output_dim)

    def forward(self, x):
        # x: [B, Input_Dim] or [B, L, Input_Dim]
        
        # Project: [B, D] or [B, L, D]
        h = self.input_proj(x)
        
        # Normalize to [B, L, D] for processing
        if h.dim() == 2:
            h = h.unsqueeze(1)
            
        # Expand to [B, L, Rate, D]
        h = h.unsqueeze(2) # [B, L, 1, D]
        h = h.repeat(1, 1, self.rate, 1) # [B, L, Rate, D]
        
        for layer in self.layers:
            h = layer(h)
            
        # Fuse branches (Sum) and flatten -> [B, L, D]
        h = h.sum(dim=2) 
        
        # Restore original rank if input was 2D
        if x.dim() == 2:
            h = h.squeeze(1) 
        
        return self.final_norm(h)

# --- mHC Components End ---


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
        else:
            raise ValueError("Unsupported RNN layer type.")
        
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor]):
        batch_size = x.size(0)
        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_size * self.chunk_size, device=x.device)
        else:
            hidden_state = hidden_state.unsqueeze(0)
        
        if self.chunk_size > 1:
            h_in = torch.chunk(hidden_state, self.chunk_size, dim=-1)
            h_in = tuple(h.contiguous() for h in h_in)
            rnn_out, h_out = self.rnn(x, h_in)
            new_hidden_state = torch.cat(h_out, dim=-1)
        else:
            h_in = hidden_state
            rnn_out, h_out = self.rnn(x, h_in)
            new_hidden_state = h_out
        
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

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = [] 
        self.hidden_states = []
        self.old_entropies = []
        self.next_value = None
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.hidden_states.clear()
        self.old_entropies.clear()
        self.next_value = None

class PPOTrainer:
    def __init__(self, config):
        self.cfg = config
        assert self.cfg.update_freq % self.cfg.seq_len == 0
        self.num_sequences = self.cfg.update_freq // self.cfg.seq_len
        assert self.num_sequences % self.cfg.batch_size == 0

        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.model = ActorCritic(state_dim, action_dim, config=self.cfg).to(self.cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.hidden_size = self.model.rnn.hidden_size * self.model.rnn.chunk_size
        
        self.step_count = 0
        self.episode_rewards = deque(maxlen=10)
        self.lr = self.cfg.lr
        self.ent_coef = self.cfg.entropy_coef
        
        print(f"Using device: {self.cfg.device}")
        print(f"Sequence length: {self.cfg.seq_len}")

    def collect_experience(self):
        self.buffer = RolloutBuffer()
        seed = random.randint(1, 2**31 - 1) if self.cfg.seed is None else self.cfg.seed
        state, _ = self.env.reset(seed=seed)
        episode_reward = 0
        
        hidden_state = torch.zeros(1, self.hidden_size, device=self.cfg.device)
        
        for _ in range(self.cfg.update_freq):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
            self.buffer.hidden_states.append(hidden_state.squeeze(0).cpu().numpy())

            with torch.no_grad():
                action, log_prob, value, next_hidden_state, predict, target, entropy = self.model.get_action(state_tensor, hidden_state)
                
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
            self.buffer.old_entropies.append(entropy)
            
            state = next_state
            hidden_state = next_hidden_state 
            self.step_count += 1
            
            if done:
                self.episode_rewards.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0
                hidden_state = torch.zeros(1, self.hidden_size, device=self.cfg.device)
                
        with torch.no_grad():
            self.buffer.next_value = self.model.get_value(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device),
                hidden_state
            ).item()

    def compute_advantages(self):
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
    
    def masked_mean(self, x, mask=None):
        if mask is None:
            return x.mean()
        else:
            assert x.shape == mask.shape
            masked_x = x * mask
            masked_count = mask.sum()
            if masked_count == 0:
                return torch.tensor(0.0, device=x.device)
            return masked_x.sum() / masked_count

    def update_model(self, advantages, returns):
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.cfg.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.cfg.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.cfg.device)
        old_entropies = torch.tensor(self.buffer.old_entropies, dtype=torch.float32).to(self.cfg.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.cfg.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32).to(self.cfg.device)
        hidden_states = torch.tensor(np.array(self.buffer.hidden_states), dtype=torch.float32).to(self.cfg.device)

        states = states.view(self.num_sequences, self.cfg.seq_len, -1)
        actions = actions.view(self.num_sequences, self.cfg.seq_len)
        old_log_probs = old_log_probs.view(self.num_sequences, self.cfg.seq_len)
        old_entropies = old_entropies.view(self.num_sequences, self.cfg.seq_len)
        advantages = advantages.view(self.num_sequences, self.cfg.seq_len)
        returns = returns.view(self.num_sequences, self.cfg.seq_len)
        old_values = old_values.view(self.num_sequences, self.cfg.seq_len)
        
        initial_hidden_states = hidden_states[::self.cfg.seq_len]

        policy_losses, value_losses, entropy_losses, rnd_losses = [], [], [], []
        approx_kls, clip_fracs, covs_list, erc_clip_fracs = [], [], [], []
        
        for _ in range(self.cfg.num_epochs):
            perm = torch.randperm(self.num_sequences, device=self.cfg.device)
            
            for start in range(0, self.num_sequences, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                idx = perm[start:end]

                s_batch = states[idx]
                a_batch = actions[idx]
                old_lp_batch = old_log_probs[idx]
                old_ent_batch = old_entropies[idx]
                adv_batch = advantages[idx]
                ret_batch = returns[idx]
                hidden_batch = initial_hidden_states[idx]
                val_batch = old_values[idx]

                logits, values, _, predict, target = self.model(s_batch, hidden_batch)
                
                logits_flat = logits.view(-1, logits.shape[-1])
                values_flat = values.view(-1)
                a_batch_flat = a_batch.view(-1)
                old_lp_batch_flat = old_lp_batch.view(-1)
                old_ent_batch_flat = old_ent_batch.view(-1)
                adv_batch_flat = adv_batch.view(-1)
                ret_batch_flat = ret_batch.view(-1)
                old_values_flat = val_batch.view(-1)

                dist = Categorical(logits=logits_flat)
                new_log_probs = dist.log_prob(a_batch_flat)
                new_entropies = dist.entropy()
                
                # ERC 计算 [cite: 122, 131, 132]
                entropy_ratio = new_entropies / (old_ent_batch_flat + 1e-8)
                erc_mask = ((entropy_ratio > (1 - self.cfg.erc_beta_low)) & 
                            (entropy_ratio < (1 + self.cfg.erc_beta_high))).float()
                
                ratio = (new_log_probs - old_lp_batch_flat).exp()
                covs = (new_log_probs - new_log_probs.mean()) * (adv_batch_flat - adv_batch_flat.mean())
                corr = torch.ones_like(adv_batch_flat) * erc_mask
                
                clip_ratio = ratio.clamp(0.0, self.cfg.dual_clip)
                surr1 = clip_ratio * adv_batch_flat
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps_min, 1 + self.cfg.clip_eps_max) * adv_batch_flat
                clip_idx = torch.where((covs > self.cfg.clip_cov_min) & (covs < self.cfg.clip_cov_max))[0]
                if len(clip_idx) > 0 and self.cfg.clip_cov_ratio > 0:
                    clip_num = max(int(len(clip_idx) * self.cfg.clip_cov_ratio), 1)
                    clip_idx = clip_idx[torch.randperm(len(clip_idx))[:min(clip_num, len(clip_idx))]]
                    corr[clip_idx] = 0.0
                
                clip_frac = self.masked_mean(
                    ((ratio < (1 - self.cfg.clip_eps_min)) | (ratio > (1 + self.cfg.clip_eps_max))).float(),
                    corr
                )
                policy_loss = self.masked_mean(-torch.min(surr1, surr2), corr)
                
                value_clip = old_values_flat + (values_flat - old_values_flat).clamp(-self.cfg.clip_eps_min, self.cfg.clip_eps_max)
                value_loss1 = (values_flat - ret_batch_flat).pow(2)
                value_loss2 = (value_clip - ret_batch_flat).pow(2)
                value_loss = 0.5 * self.masked_mean(torch.max(value_loss1, value_loss2), corr)
                
                entropy = self.masked_mean(dist.entropy(), corr)
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
                erc_clip_fracs.append(1.0 - erc_mask.mean().item())
                
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
            f"ERC Clip: {np.mean(erc_clip_fracs):.2%} | "
            f"KL: {np.mean(approx_kls):.4f} | "
            f"Clip Frac: {np.mean(clip_fracs):.2%} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Ent Coef: {self.ent_coef:.4f} | "
            f"Cov: {np.mean(covs_list):.4f}"
        )
                
    def train(self):
        while self.step_count < self.cfg.max_train_steps:
            self.collect_experience()
            advantages, returns = self.compute_advantages()
            self.update_model(advantages, returns)
            
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                print(f"Step: {self.step_count}, Avg Reward: {mean_reward:.2f}")

    def eval(self, num_episodes=10):
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
                    action, _, _, hidden_state, _, _, _ = self.model.get_action(
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
                action, _, _, hidden_state, _, _, _ = self.model.get_action(
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
        pass
    
    print("\n训练完成或被中断，开始最终测试...")
    ppo.test()