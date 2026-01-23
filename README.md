# gymRL

强化学习算法实现集合，用于学习和研究目的。
每个算法都是**独立的单文件脚本**，可以直接运行。

## 仓库结构

```
gymRL/
├── algorithms/          # 重构后的独立实现
│   ├── dqn_cartpole.py
│   ├── ppo_lunarlander.py
│   ├── sac_pendulum.py
│   └── ...
├── legacy/              # 原始实现（供参考）
├── utils/               # 共享工具（legacy代码使用）
├── assets/              # README图片
└── requirements.txt
```

## 已实现算法

### 基于价值的方法（DQN系列）

| 文件 | 算法 | 环境 | 关键特性 |
|------|------|------|----------|
| `dqn_cartpole.py` | DQN | CartPole-v1 | 经验回放、目标网络 |
| `ddqn_per_cartpole.py` | Double DQN + PER | CartPole-v1 | 双Q学习、优先经验回放 |
| `ddqn_per_duel_cartpole.py` | DDQN + PER + Dueling | CartPole-v1 | Dueling架构、PER |
| `noisy_dqn_cartpole.py` | NoisyNet DQN | CartPole-v1 | 噪声层探索 |
| `rainbow_dqn_cartpole.py` | Rainbow DQN | CartPole-v1 | NoisyNet、Dueling、PER、N-step |
| `noisy_dqn_flappybird.py` | NoisyNet DQN | FlappyBird-v0 | PSCN骨干网络、Dueling |

### 策略梯度方法（PPO系列）

| 文件 | 算法 | 环境 | 关键特性 |
|------|------|------|----------|
| `ppo_lunarlander.py` | PPO | LunarLander-v3 | 裁剪替代目标、GAE、dual-clip |
| `ppo_rnn_lunarlander.py` | PPO + RNN | LunarLander-v3 | 基于GRU的Actor-Critic |
| `ppo_rnn_flappybird.py` | PPO + RNN | FlappyBird-v0 | PSCN + GRU骨干网络 |
| `ppg_rnn_lunarlander.py` | PPG + RNN | LunarLander-v3 | 相位策略梯度 |
| `ppo_full_lunarlander.py` | PPO (完整版) | LunarLander-v3 | mHC、ERC、全部技巧 |
| `ppo_lstm_lunarlander.py` | PPO + LSTM | LunarLander-v3 | RND、mHC、LSTM |

### Actor-Critic方法（连续控制）

| 文件 | 算法 | 环境 | 关键特性 |
|------|------|------|----------|
| `ddpg_pendulum.py` | DDPG | Pendulum-v1 | 确定性策略、软更新 |
| `td3_pendulum.py` | TD3 | Pendulum-v1 | 双Critic、延迟更新 |
| `sac_pendulum.py` | SAC | Pendulum-v1 | 熵正则化、自动alpha调节 |
| `sac_cartpole.py` | SAC (离散) | CartPole-v1 | 离散动作SAC |

### 经典强化学习（表格方法）

| 文件 | 算法 | 环境 | 关键特性 |
|------|------|------|----------|
| `qlearning_frozenlake.py` | Q-Learning | FrozenLake-v1 | Q表、奖励塑形 |
| `qlearning_cliffwalking.py` | Q-Learning | CliffWalking-v0 | ε-贪婪探索 |
| `mountaincar_baseline.py` | 规则策略 | MountainCar-v0 | 手工设计的基线 |

## 快速开始

每个算法文件都是独立的，可以直接运行：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行任意算法
python algorithms/dqn_cartpole.py
python algorithms/ppo_lunarlander.py
python algorithms/td3_pendulum.py
```

训练过程中按 `Ctrl+C` 可以优雅地停止并运行评估。

## 推荐算法

- **离散动作（入门）**：从 `dqn_cartpole.py` 开始
- **离散动作（进阶）**：使用 `ppo_full_lunarlander.py` 或 `ppo_lstm_lunarlander.py`
- **连续动作**：使用 `td3_pendulum.py`（最稳定）
- **Rainbow DQN**：使用 `rainbow_dqn_cartpole.py` 体验DQN全部改进

## PPO技巧实现

`ppo_full_lunarlander.py` 和 `ppo_lstm_lunarlander.py` 包含现代强化学习技巧：

- `value_clip` - 裁剪价值函数更新
- `clip-higher` - 非对称裁剪边界
- `dual-clip` - 负优势的双重裁剪
- `ent_coef` - 熵正则化与退火
- `decouple-lambda` - Actor/Critic分离的GAE lambda
- `ERC` - 熵比率裁剪
- `RND` - 随机网络蒸馏（探索）
- `mHC` - 流形超连接网络架构
- `adam-eps` - 调优的Adam epsilon参数

## Tensorboard日志

部分算法支持Tensorboard日志：

```bash
tensorboard --logdir=exp
```

然后打开 http://localhost:6006/ 查看训练曲线。

![tensorboard示例](assets/image-20240602232200101.png)

## 可视化训练

要实时观看训练过程，修改Config类：

```python
class Config:
    def __init__(self):
        self.render_mode = 'human'  # 默认是 'rgb_array'
```

![训练可视化](assets/image-20240602231618885.png)

## 算法简介

### PPO（Proximal Policy Optimization）

PPO是一种策略优化算法，通过限制每次策略更新的幅度来稳定训练过程。

### DQN（Deep Q-Network）

DQN使用深度神经网络来逼近Q值函数，并结合经验回放和固定Q目标来稳定训练。

### SAC（Soft Actor-Critic）

SAC是一种最大化策略熵的算法，旨在提高策略的探索能力。

### DDPG（Deep Deterministic Policy Gradient）

DDPG是一种结合了策略梯度和Q学习的算法，适用于连续动作空间。

### TD3（Twin Delayed DDPG）

TD3是对DDPG的改进，通过延迟更新策略网络和目标网络来减少Q值的过估计。

## 许可证

本项目采用MIT许可证，详见LICENSE。
