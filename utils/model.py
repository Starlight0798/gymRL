from torch import nn
import torch
import numpy as np
import math
from torch.nn import functional as F


# 正交初始化
def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


# 全连接层
class MLP(nn.Module):
    def __init__(self,
                 dim_list,
                 activation=nn.ReLU(),
                 last_act=False,
                 use_norm=False,
                 linear=nn.Linear
                 ):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dim_list) - 1):
            layer = orthogonal_init(linear(dim_list[i], dim_list[i + 1]))
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


# 深度可分离卷积层
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 卷积层
class ConvBlock(nn.Module):
    def __init__(self,
                 channels: list[tuple],
                 kernel_size=3,
                 stride=1,
                 padding=2,
                 use_depthwise=True,
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
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU())
            self.conv_layers.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        return self.conv_layers(x)


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
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = MLP([d_model, d_model, d_model], "encoder_ffn")

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.ln1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.ln2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(d_model, n_heads)
        self.attn2 = MultiHeadAttention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = MLP([d_model, d_model, d_model], "decoder_ffn")
        
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = x + self.dropout(self.attn1(x, x, x, tgt_mask))
        x = self.ln1(x)
        x = x + self.dropout(self.attn2(x, enc_out, enc_out, src_mask))
        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        x = self.ln3(x)
        return x


class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PSCN, self).__init__()
        assert output_dim > 0 and (output_dim & (output_dim - 1) == 0), "output_dim must be a power of 2"
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



if __name__ == "__main__":
    # 测试多头注意力机制
    x = torch.randn(32, 512)
    mha = MultiHeadAttention(512, 8)
    y = mha(x, x, x)
    print(y.shape)
