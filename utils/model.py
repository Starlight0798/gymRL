from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import os
from utils.buffer import Queue
from loguru import logger

def initialize_weights(layer, init_type='kaiming', nonlinearity='leaky_relu'):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if init_type == 'kaiming':                  # kaiming初始化，适合激活函数为ReLU, LeakyReLU, PReLU
            nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(layer.weight)   # xavier初始化, 适合激活函数为tanh和sigmoid
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))       # 正交初始化，适合激活函数为ReLU
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
                 linear=nn.Linear,
                 *args, **kwargs
                 ):
        super(MLP, self).__init__()
        assert dim_list, "Dim list can't be empty!"
        layers = []
        for i in range(len(dim_list) - 1):
            layer = initialize_weights(linear(dim_list[i], dim_list[i + 1], *args, **kwargs))
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
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features), persistent=False)

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features), persistent=False)

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))  
        
    def scale_noise(self, size: int):
        x = torch.randn(size)  
        x = x.sign().mul(x.abs().sqrt())
        return x
    

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)
        

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, sigma_init={self.sigma_init})"


# 深度可分离卷积层，参数更少，效率比Conv2d更高
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
    
# 带噪声的卷积层
class NoisyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, sigma_init=0.5):
        super(NoisyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sigma_init = sigma_init
        self.groups = groups

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_channels, in_channels // groups, kernel_size, kernel_size))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_channels))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_channels))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_channels))

        self.reset_parameters()
        self.reset_noise()


    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, groups=self.groups)


    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_channels * self.kernel_size * self.kernel_size)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_channels * self.kernel_size * self.kernel_size))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_channels))


    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


    def reset_noise(self):
        epsilon_weight = self.scale_noise((self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        epsilon_bias = self.scale_noise(self.out_channels)
        self.weight_epsilon.copy_(epsilon_weight)
        self.bias_epsilon.copy_(epsilon_bias)


    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, groups={self.groups}, sigma_init={self.sigma_init})"

    

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # Create a position index (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Create a dimension index (0, 1, 2, ..., d_model/2-1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # Apply sine to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a new dimension to make the shape (1, max_len, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register the positional encoding matrix as a buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input tensor
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size needs to be divisible by num_heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.num_heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        # Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_dim)
        
        out = self.fc_out(out)
        return out
    
    

# 一种兼顾宽度和深度的全连接层，提取信息效率更高
class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim, depth=4, linear=nn.Linear):
        super(PSCN, self).__init__()
        min_dim = 2 ** (depth - 1)
        assert depth >= 1, "depth must be at least 1"
        assert output_dim >= min_dim, f"output_dim must be >= {min_dim} for depth {depth}"
        assert output_dim % min_dim == 0, f"output_dim must be divisible by {min_dim} for depth {depth}"
        
        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        in_dim, out_dim = input_dim, output_dim
        
        for i in range(depth):
            self.layers.append(MLP([in_dim, out_dim], last_act=True, linear=linear))
            in_dim = out_dim // 2
            out_dim //= 2 

    def forward(self, x):
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


# 将MLP和RNN以3:1的比例融合
class MLPRNN(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super(MLPRNN, self).__init__()
        assert output_dim % 4 == 0, "output_dim must be divisible by 4"
        self.rnn_size = output_dim // 4
        self.rnn_linear = MLP([input_dim, 3 * self.rnn_size])
        self.rnn = nn.GRU(input_dim, self.rnn_size, *args, **kwargs)

    def forward(self, x, rnn_state: torch.Tensor):
        rnn_linear_out = self.rnn_linear(x)
        rnn_out, rnn_state = self.rnn(x, rnn_state)
        out = torch.cat([rnn_linear_out, rnn_out], dim=-1)
        return out, rnn_state
    


# 循环神经网络基类，覆盖基本方法
class BaseRNNModel(nn.Module):
    def __init__(self, device, hidden_size):
        super(BaseRNNModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.reset_hidden()

    @torch.jit.export
    def reset_hidden(self):
        self.rnn_h = torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float)
        
    @torch.jit.export
    def get_hidden(self):
        return self.rnn_h
    
    @torch.jit.export
    def set_hidden(self, hidden):
        self.rnn_h = hidden




# 管理模型加载与存储
class ModelLoader:
    def __init__(self, cfg):
        cfg.save_path = f'./checkpoints/{cfg.algo_name}_{cfg.env_name.replace("/", "-")}.pth'
        self.cfg = cfg
        if not os.path.exists(os.path.dirname(cfg.save_path)):
            os.makedirs(os.path.dirname(cfg.save_path))

    def save_model(self):
        state = {}
        for key, value in self.__dict__.items():
            exclude_keys = ['state_buffer', 'cfg', 'memory']
            if key in exclude_keys:
                continue
            logger.debug(f"Save {key}")
            if hasattr(value, 'state_dict'):
                state[f'{key}_state_dict'] = value.state_dict()
            else:
                state[key] = value
        torch.save(state, self.cfg.save_path)
        self._print_model_summary()
        logger.info(f"Save model to {self.cfg.save_path}")

    def load_model(self):
        with logger.catch(message="Model loading failed."):
            checkpoint = torch.load(self.cfg.save_path, map_location=self.cfg.device)
            for key, value in checkpoint.items():
                exclude_keys = ['state_buffer', 'cfg', 'memory']
                if key in exclude_keys:
                    continue
                logger.debug(f"Load {key}")
                if key.endswith('_state_dict'):
                    attr_name = key.replace('_state_dict', '')
                    if hasattr(self, attr_name):
                        getattr(self, attr_name).load_state_dict(value)
                else:
                    setattr(self, key, value)
            logger.info(f"Load model： {self.cfg.save_path}")
            
            
    def _print_model_summary(self):
        if hasattr(self, 'net'):
            num_params = sum(p.numel() for p in self.net.parameters())
            message = f"Model Summary: Number of parameters: {num_params}\n"
            for name, param in self.net.named_parameters():
                message += f"{name}: {param.numel()} parameters\n"
            logger.debug(message)
                

class StateManager:
    def __init__(self, buffer_size=100):
        self.state_buffer = Queue(buffer_size)

    def save_state(self, *args):
        self.state_buffer.put(args)

    def load_state(self):
        return self.state_buffer.sample()
    
