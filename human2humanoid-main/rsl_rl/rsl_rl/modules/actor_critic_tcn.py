# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 版权声明和许可证信息保持不变

import torch
import torch.nn as nn
from torch.distributions import Normal

class TemporalBlock(nn.Module):
    """TCN的基本构建块，包含扩张因果卷积和残差连接"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.1):
        super().__init__()
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=0, dilation=dilation)
        # 计算左侧padding以保持因果性
        self.padding1 = (kernel_size - 1) * dilation
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=stride, padding=0, dilation=dilation)
        self.padding2 = (kernel_size - 1) * dilation
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 如果输入输出通道数不同，使用1x1卷积调整维度
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.dropout_rate = dropout

    def forward(self, x):
        # 保存残差连接
        residual = x
        
        # 第一层卷积
        out = nn.functional.pad(x, (self.padding1, 0))  # 左侧padding
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # 第二层卷积
        out = nn.functional.pad(out, (self.padding2, 0))  # 左侧padding
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # 处理残差连接
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        return out

class TCN(nn.Module):
    """完整的TCN网络"""
    def __init__(self, input_dim, output_dim, context_len, 
                 num_channels=128, kernel_size=3, dropout=0.1, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_len = context_len
        
        # 输入层
        self.input_layer = nn.Linear(input_dim, num_channels)
        
        # 创建TCN层
        layers = []
        dilation_size = 1
        for i in range(num_layers):
            layers.append(
                TemporalBlock(num_channels, num_channels, kernel_size, 
                            stride=1, dilation=dilation_size, dropout=dropout)
            )
            dilation_size *= 2  # 指数增长的扩张因子
        
        self.tcn = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, output_dim)
        )
    
    def forward(self, x):
        # x形状: (batch_size, context_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 通过输入层
        x = self.input_layer(x)  # (batch_size, seq_len, num_channels)
        
        # 转换为卷积需要的形状 (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        
        # 通过TCN层
        x = self.tcn(x)
        
        # 取最后一个时间步
        x = x[:, :, -1]  # (batch_size, num_channels)
        
        # 通过输出层
        x = self.output_layer(x)
        
        return x

class ActorCriticTCN(nn.Module):
    """基于TCN的Actor-Critic网络"""
    is_recurrent = False
    
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, 
                 obs_context_len, init_noise_std=1.0, **kwargs):
        if kwargs:
            print("ActorCriticTCN.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        # 策略网络 (Actor)
        self.actor = TCN(num_actor_obs, num_actions, obs_context_len)
        self.actor.output_layer[1].weight.data *= 0.01  # 初始化最后一层权重较小

        # 价值函数 (Critic)
        self.critic = TCN(num_critic_obs, 1, obs_context_len)

        print(f"Actor TCN: {self.actor}")
        print(f"Critic TCN: {self.critic}")

        # 动作噪声
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # 禁用参数验证以提高速度
        Normal.set_default_validate_args = False

    # 以下方法与原始实现保持不变
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value