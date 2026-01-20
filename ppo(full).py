import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import gymnasium as gym
from collections import deque
import warnings
import signal
import sys
from typing import List, Type, Optional
import math
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
        self.batch_size = 1024          # 每次更新的批次大小
        self.gamma = 0.995              # 折扣因子
        self.lam_actor = 0.95           # GAE参数 - actor
        self.lam_critic = 0.95          # GAE参数 - critic
        self.clip_eps_min = 0.2         # PPO-CLIP-MIN参数 
        self.clip_eps_max = 0.28        # PPO-CLIP-MAX参数
        self.clip_cov_ratio = 0.0       # PPO-COV-RATIO参数
        self.clip_cov_min = 1.0         # PPO-COV-MIN参数
        self.clip_cov_max = 5.0         # PPO-COV-MAX参数
        self.dual_clip = 3.0            # 双重裁剪
        self.entropy_coef = 0.01        # 熵奖励系数
        self.erc_beta_low = 0.06        # ERC 下限参数
        self.erc_beta_high = 0.06       # ERC 上限参数
        self.lr = 3e-4                  # 学习率
        self.max_grad_norm = 0.5        # 梯度裁剪阈值
        self.anneal = True              # 是否退火
        self.device = 'cpu'

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
        # x: [B, Input_Dim]
        B = x.shape[0]
        
        # Project and expand to [B, 1, Rate, D]
        h = self.input_proj(x)
        h = h.unsqueeze(1).unsqueeze(2)
        h = h.repeat(1, 1, self.rate, 1)
        
        for layer in self.layers:
            h = layer(h)
            
        # Fuse branches (Sum) and flatten
        h = h.sum(dim=2) 
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

# 演员-评论家网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config=None):
        super().__init__()
        
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
            # 共享网络层
            self.shared = PSCN(
                input_dim=state_dim, 
                output_dim=256,
                depth=4,
            )
            shared_out_dim = 256

        # 策略头
        self.actor = MLP([shared_out_dim, 256, action_dim], last_std=0.001)
        # 价值头
        self.critic = MLP([shared_out_dim, 256, 1], last_std=1.0)
            
    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)
    
    def get_action(self, x, deterministic=False):
        """采样动作并返回相关数据"""
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample() if not deterministic else logits.argmax(dim=-1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.cpu().item(), log_prob.cpu().item(), value.squeeze().cpu(), entropy.cpu().item()
    
    def get_value(self, x):
        """获取状态价值"""
        _, value = self.forward(x)
        return value.squeeze().cpu()

# 经验回放缓存
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = [] 
        self.old_entropies = []
        self.next_value = None
        
    def clear(self):
        """清空缓存"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.old_entropies.clear()
        self.next_value = None

# PPO训练器
class PPOTrainer:
    def __init__(self, config):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        # 初始化模型和优化器
        self.model = ActorCritic(state_dim, action_dim, config=self.cfg).to(self.cfg.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.cfg.lr, 
            eps=1e-5
        )
        
        # 训练状态跟踪
        self.step_count = 0
        self.episode_rewards = deque(maxlen=10)
        self.lr = self.cfg.lr
        self.ent_coef = self.cfg.entropy_coef
        
        # 打印模型设备
        print(f"Model device: {next(self.model.parameters()).device}")

    def collect_experience(self):
        """收集训练经验数据"""
        self.buffer = RolloutBuffer()
        seed = random.randint(1, 2**31 - 1) if self.cfg.seed is None else self.cfg.seed
        state, _ = self.env.reset(seed=seed)
        episode_reward = 0
        
        for _ in range(self.cfg.update_freq):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
            with torch.no_grad():
                action, log_prob, value, entropy = self.model.get_action(state_tensor)
                
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # 存储经验数据
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(log_prob)
            self.buffer.values.append(value)
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)
            self.buffer.old_entropies.append(entropy)
            
            state = next_state
            episode_reward += reward
            self.step_count += 1
            
            if done:
                self.episode_rewards.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0
                
        # 获取最后一步的状态价值
        with torch.no_grad():
            self.buffer.next_value = self.model.get_value(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)).item()

    def compute_advantages(self):
        """计算GAE优势估计"""
        rewards = np.array(self.buffer.rewards)
        dones = np.array(self.buffer.dones)
        values = np.array(self.buffer.values + [self.buffer.next_value])
        
        # GAE计算
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
        """执行PPO参数更新"""
        states = torch.tensor(self.buffer.states, dtype=torch.float32).to(self.cfg.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.float32).to(self.cfg.device).long()
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.cfg.device)
        old_entropies = torch.tensor(self.buffer.old_entropies, dtype=torch.float32).to(self.cfg.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.cfg.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, old_entropies, advantages, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        
        # 训练指标跟踪
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fracs = []
        covs_list = []
        erc_clip_fracs = []
        
        for _ in range(self.cfg.num_epochs):
            for batch in loader:
                s_batch, a_batch, old_lp_batch, old_ent_batch, adv_batch, ret_batch = batch

                # 计算新策略的输出
                logits, values = self.model(s_batch)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(a_batch)
                new_entropies = dist.entropy()
                
                # 计算熵比例 (Entropy Ratio)
                entropy_ratio = new_entropies / (old_ent_batch + 1e-8)
                erc_mask = ((entropy_ratio > (1 - self.cfg.erc_beta_low)) & 
                            (entropy_ratio < (1 + self.cfg.erc_beta_high))).float()
                
                # 重要性采样比率
                ratio = (new_log_probs - old_lp_batch).exp()
                covs = (new_log_probs - new_log_probs.mean()) * (adv_batch - adv_batch.mean())
                corr = torch.ones_like(adv_batch) * erc_mask
                
                # 策略损失计算  
                clip_ratio = ratio.clamp(0.0, self.cfg.dual_clip)
                surr1 = clip_ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps_min, 1 + self.cfg.clip_eps_max) * adv_batch
                clip_idx = torch.where((covs > self.cfg.clip_cov_min) & (covs < self.cfg.clip_cov_max))[0]
                if len(clip_idx) > 0 and self.cfg.clip_cov_ratio > 0:
                    clip_num = max(int(len(clip_idx) * self.cfg.clip_cov_ratio), 1)
                    clip_idx = clip_idx[torch.randperm(len(clip_idx))[:min(clip_num, len(clip_idx))]]
                    corr[clip_idx] = 0.0
                clip_frac = torch.mean(((ratio < (1 - self.cfg.clip_eps_min)) | (ratio > (1 + self.cfg.clip_eps_max))).float() * corr)
                policy_loss = torch.mean(-torch.min(surr1, surr2) * corr)
                
                # 值函数损失计算
                value_loss = torch.mean(0.5 * corr * (values.squeeze() - ret_batch).pow(2))
                
                # 熵正则项
                entropy = (dist.entropy() * corr).mean()
                entropy_loss = torch.mean(-self.ent_coef * entropy)
                
                # 总损失计算
                loss = policy_loss + value_loss + entropy_loss
                
                # 参数更新步骤
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                
                # 记录训练指标
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                clip_fracs.append(clip_frac.item())
                covs_list.append(covs.mean().item())
                erc_clip_fracs.append(1.0 - erc_mask.mean().item())
                
                # 计算近似KL散度
                with torch.no_grad():
                    approx_kl = (old_lp_batch - new_log_probs).mean().item()
                    approx_kls.append(approx_kl)
        
        # 学习率退火 
        if self.cfg.anneal:
            self.lr = self.cfg.lr * (1 - self.step_count / self.cfg.max_train_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            self.ent_coef = self.cfg.entropy_coef * (1 - self.step_count / self.cfg.max_train_steps)
        
        # 输出训练信息
        print(
            f"Policy Loss: {np.mean(policy_losses):.4f} | "
            f"Value Loss: {np.mean(value_losses):.4f} | "
            f"Entropy: {np.mean(entropy_losses):.4f} | "
            f"KL: {np.mean(approx_kls):.4f} | "
            f"Clip Frac: {np.mean(clip_fracs):.2%} | "
            f"ERC Clip: {np.mean(erc_clip_fracs):.2%} | "
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
            
            # 输出训练进度
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
            while not done:
                with torch.no_grad():
                    action, _, _, _ = self.model.get_action(torch.tensor(state, dtype=torch.float32).to(self.cfg.device), deterministic=True)
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
        env.render()
        done = False
        total_reward = 0
        
        while not done:
            env.render()
            with torch.no_grad():
                action, _, _, _ = self.model.get_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device), deterministic=True)
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
        print("\n训练被中断，开始测试...")
        ppo.test()
    else:
        ppo.test()