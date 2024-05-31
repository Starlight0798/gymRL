from torch import nn
import torch
import numpy as np
import math
from torch.nn import functional as F
import os
from utils.buffer import Queue

def initialize_weights(layer, init_type='kaiming', nonlinearity='leaky_relu'):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if init_type == 'kaiming':                  # kaiming初始化，适合激活函数为ReLU, LeakyReLU, PReLU
            nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(layer.weight)   # xavier初始化, 适合激活函数为tanh和sigmoid
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(layer.weight, gain=sqrt(2))       # 正交初始化，适合激活函数为ReLU
        else:       
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


# 全连接层
class MLP(nn.Module):
    def __init__(self,
                 dim_list,
                 activation=nn.PReLU(),
                 last_act=False,
                 use_norm=False,
                 linear=nn.Linear
                 ):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dim_list) - 1):
            layer = initialize_weights(linear(dim_list[i], dim_list[i + 1]))
            layers.append(layer)
            if i < len(dim_list) - 2:
                if use_norm:
                    layers.append(nn.LayerNorm(dim_list[i + 1]))
                layers.append(activation)
        if last_act:
            if use_norm:
                layers.append(nn.LayerNorm(dim_list[-1]))
            layers.append(activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 带噪声的全连接层
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))  # 这里要除以out_features

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)  # torch.randn产生标准高斯分布
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)


# 深度可分离卷积层，参数更少，效率比Conv2d更高
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 卷积网络块
class ConvBlock(nn.Module):
    def __init__(self,
                 channels: list[tuple],
                 output_dim,
                 input_shape=(3, 84, 84),
                 kernel_size=3,
                 stride=1,
                 padding=2,
                 use_depthwise=False,
                 activation=nn.PReLU()
                 ):
        super(ConvBlock, self).__init__()
        self.conv_layers = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(channels):
            if use_depthwise:
                self.conv_layers.add_module(f'conv_dw_{i}', DepthwiseSeparableConv(in_channels,
                                                                                   out_channels,
                                                                                   kernel_size,
                                                                                   stride,
                                                                                   padding))
            else:
                self.conv_layers.add_module(f'conv_{i}', nn.Conv2d(in_channels,
                                                                   out_channels,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding))
            self.conv_layers.add_module(f'bn_{i}', nn.BatchNorm2d(out_channels))
            self.conv_layers.add_module(f'act_{i}', activation)
            self.conv_layers.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(2, 2)))
            
        self.output_dim = output_dim
        self._initialize_fc(input_shape, channels)

    def _initialize_fc(self, input_shape, channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = dummy_input
            for layer in self.conv_layers:
                x = layer(x)
            n_features = x.size(1) * x.size(2) * x.size(3)
            self.fc = MLP([n_features, self.output_dim])
            print(f'卷积输出维度：{n_features}')


    def forward(self, x):
        features = self.conv_layers(x)
        flat = torch.flatten(features, 1) 
        out = self.fc(flat)
        return out


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads, 
                 d_k=None, 
                 d_v=None, 
                 out=None,
                 act=nn.ReLU(),
                 last_act=True,
                 dropout=0.1,
                 ):
        super(MultiHeadAttention, self).__init__()
        if d_k is None or d_v is None:
            assert d_model % n_heads == 0
            self.d_k = self.d_v = d_model // n_heads
        else:
            self.d_k = d_k
            self.d_v = d_v
        self.out_dim = d_model if out is None else out
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.Q = MLP([d_model, n_heads * self.d_k], act)
        self.K = MLP([d_model, n_heads * self.d_k], act)
        self.V = MLP([d_model, n_heads * self.d_v], act)
        if last_act:
            self.out = MLP([n_heads * self.d_v, self.out_dim], act, last_act=True)
        else:
            self.out = MLP([n_heads * self.d_v, self.out_dim], act)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.Q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.K(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.V(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, mask)
        concat = (scores.transpose(1, 2).contiguous()
                  .view(batch_size, self.n_heads * self.d_v))
        output = self.out(concat)
        return output

    def attention(self, q, k, v, d_k, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output
    
# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = MLP([d_model, d_model, d_model])

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.ln1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.ln2(x)
        return x

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(d_model, n_heads)
        self.attn2 = MultiHeadAttention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = MLP([d_model, d_model, d_model])
        
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = x + self.dropout(self.attn1(x, x, x, tgt_mask))
        x = self.ln1(x)
        x = x + self.dropout(self.attn2(x, enc_out, enc_out, src_mask))
        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        x = self.ln3(x)
        return x

# 一种兼顾宽度和深度的全连接层，提取信息效率更高
class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PSCN, self).__init__()
        assert output_dim >= 32 and output_dim % 8 == 0, "output_dim must be >= 32 and divisible by 8"
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

# 将MLP和RNN以3:1的比例融合
class MLPRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn=nn.GRU, *args, **kwargs):
        super(MLPRNN, self).__init__()
        assert output_dim % 4 == 0, "output_dim must be divisible by 4"
        self.rnn_size = output_dim // 4
        self.rnn_linear = MLP([input_dim, 3 * self.rnn_size])
        self.rnn = rnn(input_dim, self.rnn_size, *args, **kwargs)

    def forward(self, x, rnn_state: torch.Tensor):
        rnn_linear_out = self.rnn_linear(x)
        rnn_out, rnn_state = self.rnn(x, rnn_state)
        out = torch.cat([rnn_linear_out, rnn_out], dim=1)
        return out, rnn_state
    

# 循环神经网络基类，覆盖基本方法
class BaseRNNModel(nn.Module):
    def __init__(self, device, hidden_size):
        super(BaseRNNModel, self).__init__()
        self.device = device
        self.rnn_h = torch.zeros(1, hidden_size, device=self.device, dtype=torch.float)

    @torch.jit.export
    def reset_hidden(self):
        self.rnn_h = torch.zeros_like(self.rnn_h, device=self.device, dtype=torch.float)
        
    @torch.jit.export
    def get_hidden(self):
        return self.rnn_h
    
    @torch.jit.export
    def set_hidden(self, hidden):
        self.rnn_h = hidden
    

# convmixer使用的层
class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size = 9):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = kernel_size, groups = dim, padding = 'same'),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x


# 残差卷积网络块
class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size = 9, patch_size = 7, output = 512):
        super().__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvMixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim = dim, kernel_size = kernel_size))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, output)
        )
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.conv2d1(x)
        for ConvMixer_block in self.ConvMixer_blocks:
            x = ConvMixer_block(x)
        x = self.head(x)
        return x


# 管理模型加载与存储
class ModelLoader:
    def __init__(self, cfg):
        cfg.save_path = f'./checkpoints/{cfg.algo_name}_{cfg.env_name}.pth'
        self.cfg = cfg
        if not os.path.exists(os.path.dirname(cfg.save_path)):
            os.makedirs(os.path.dirname(cfg.save_path))

    def save_model(self):
        state = {}
        for key, value in self.__dict__.items():
            exclude_keys = ['state_buffer', 'cfg', 'memory']
            if key in exclude_keys:
                continue
            if hasattr(value, 'state_dict'):
                state[f'{key}_state_dict'] = value.state_dict()
            else:
                state[key] = value
        torch.save(state, self.cfg.save_path)
        self._print_model_summary()
        print(f"模型保存到 {self.cfg.save_path}")

    def load_model(self):
        try:
            checkpoint = torch.load(self.cfg.save_path, map_location=self.cfg.device)
            for key, value in checkpoint.items():
                exclude_keys = ['state_buffer', 'cfg', 'memory']
                if key in exclude_keys:
                    continue
                if key.endswith('_state_dict'):
                    attr_name = key.replace('_state_dict', '')
                    if hasattr(self, attr_name):
                        getattr(self, attr_name).load_state_dict(value)
                else:
                    setattr(self, key, value)
            print(f"模型加载： {self.cfg.save_path}")
        except FileNotFoundError as e:
            print(f'模型加载失败：{str(e)}')   
            
    def _print_model_summary(self):
        if hasattr(self, 'model'):
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model Summary: Number of parameters: {num_params}")
            for name, param in self.model.named_parameters():
                print(f"{name}: {param.numel()} parameters")
                

class StateManager:
    def __init__(self, buffer_size=100):
        self.state_buffer = Queue(buffer_size)

    def save_state(self, *args):
        self.state_buffer.put(args)

    def load_state(self):
        return self.state_buffer.sample()
    
