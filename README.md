# gymRL
 本人学习强化学习(PPO,DQN,SAC,DDPG等算法)，在gym环境下写的代码集。

 主要研究了PPO和DQN类算法，根据各个论文复现了如下改进:

-  PPO: dual-PPO, clip-PPO, use-RNN, attention etc.
-  DQN: rainbow DQN

在离散动作空间和连续动作空间，PPO算法都奏效，因此我对离散和连续动作空间的代码探索主要在CartPole(PPO), Pendulum(PPO)两份代码进行，读者可以重点关注。
感谢知乎分享：https://zhuanlan.zhihu.com/p/654972230 提出的神经网络构建思路，尤其是其提出的PSCN层：

```python
class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PSCN, self).__init__()
        assert output_dim >= 32 and output_dim % 8 == 0, "output_dim must be >= 32 and divisible by 8 "
        self.hidden_dim = output_dim
        self.fc1 = MLP([input_dim, self.hidden_dim], last_act=True)
        self.fc2 = MLP([self.hidden_dim // 2, self.hidden_dim // 2], last_act=True)
        self.fc3 = MLP([self.hidden_dim // 4, self.hidden_dim // 4], last_act=True)
        self.fc4 = MLP([self.hidden_dim // 8, self.hidden_dim // 8], last_act=True)

    def forward(self, x):
        x = self.fc1(x)

        x1 = x[:, :self.hidden_dim // 2]
        x = x[:, self.hidden_dim // 2:]
        x = self.fc2(x)

        x2 = x[:, :self.hidden_dim // 4]
        x = x[:, self.hidden_dim // 4:]
        x = self.fc3(x)

        x3 = x[:, :self.hidden_dim // 8]
        x = x[:, self.hidden_dim // 8:]
        x4 = self.fc4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out
    
# MLP层即封装的全连接层，具体实现请参照utils/model.py
```

个人在这个层的实践过程中效果颇好，读者可以斟酌使用。
