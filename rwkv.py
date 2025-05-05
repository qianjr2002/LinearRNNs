#! -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import math

class RWKV(nn.Module):
    """RWKV模块 - PyTorch实现
    链接1 https://github.com/BlinkDL/RWKV-LM
    链接2 https://kexue.fm/archives/9554
    
    Args:
        hidden_size: 隐藏层大小
        units: 输出单元数
        use_bias: 是否使用偏置 默认为True
        unroll: 是否展开计算，可加速训练但会增加显存消耗
    """
    def __init__(
        self,
        hidden_size,
        units,
        use_bias=True,
        unroll=True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.units = units
        self.use_bias = use_bias
        self.unroll = unroll
        
        # RKV映射层
        self.rkv_dense = nn.Linear(hidden_size, units * 3, bias=use_bias)
        # 输出映射层
        self.o_dense = nn.Linear(units, hidden_size, bias=use_bias)
        
        # 初始化参数
        r_min, r_max = 0.9, 0.999
        u = torch.rand(units)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u * (r_max**2 - r_min**2) + r_min**2))
        )
        self.gamma_log = nn.Parameter(torch.zeros(units))

    def forward(self, inputs):
        rkv = self.rkv_dense(inputs)
        r, k, v = torch.chunk(rkv, 3, dim=-1)
        r = torch.sigmoid(r)
        k = torch.exp(k)
        
        kv = k * v
        u = torch.cat([kv, k], dim=-1)
        nu = torch.exp(torch.cat([self.nu_log, self.nu_log]))
        gamma = torch.exp(self.nu_log + self.gamma_log) - 1

        B, L, _ = inputs.shape
        
        # 计算需要的填充长度
        log2_L = int(np.ceil(np.log2(L)))
        pad_len = 2**log2_L - L
        u_padded = torch.nn.functional.pad(u, (0, 0, 0, pad_len))

        def rwkv_step(x, i):
            l = 2**i
            x = x.view(B * (2**log2_L) // l, l, -1)
            x1, x2 = x[:, :l//2], x[:, l//2:]
            
            pos = torch.arange(1, l//2 + 1, device=x.device, dtype=torch.float32)
            nus = torch.outer(pos, nu)
            lambs = torch.exp(-nus)
            
            x2 = x2 + lambs * x1[:, -1:]
            x = torch.cat([x1, x2], dim=1)
            return x.view(B, 2**log2_L, -1)

        x = u_padded
        if self.unroll:
            for i in range(log2_L):
                x = rwkv_step(x, i + 1)
        else:
            for i in range(1, log2_L + 1):
                x = rwkv_step(x, i)

        x = x[:, :L]
        u1, u2 = torch.chunk(x, 2, dim=-1)
        x = (u1 + gamma * kv) / (u2 + gamma * k + 1e-6) * r
        
        return self.o_dense(x)
