import torch
import torch.nn as nn
from lru import LRU
from slru import SLRU
from rwkv import RWKV

class RNNWrapper(nn.Module):
    """RNN包装类，用于计算参数量和FLOPs
    
    Args:
        rnn_type: 使用的RNN类型 ('lru', 'slru', 或 'rwkv')
        hidden_size: 隐藏层大小
        freq_size: 频率维度大小
    """
    def __init__(self, rnn_type='lru', hidden_size=64, freq_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.freq_size = freq_size
        
        # 选择RNN类型
        rnn_classes = {
            'lru': LRU,
            'slru': SLRU,
            'rwkv': RWKV
        }
        rnn_class = rnn_classes[rnn_type.lower()]
        
        # 为每个频率创建一个RNN
        self.rnn_layers = nn.ModuleList([
            rnn_class(hidden_size=hidden_size, units=hidden_size)
            for _ in range(freq_size)
        ])

    def forward(self, x):
        # 输入shape: (B, C, F, T)
        B, C, F, T = x.shape
        assert C == self.hidden_size and F == self.freq_size
        
        # 重排维度以便处理
        x = x.permute(0, 2, 1, 3)  # (B, F, C, T)
        x = x.reshape(B * F, C, T)  # (B*F, C, T)
        x = x.permute(0, 2, 1)     # (B*F, T, C)
        
        # 对每个频率分别处理
        outputs = []
        for i in range(F):
            batch_start = i * B
            batch_end = (i + 1) * B
            output = self.rnn_layers[i](x[batch_start:batch_end])  # (B, T, C)
            outputs.append(output)
            
        # 重组输出
        x = torch.stack(outputs, dim=1)  # (B, F, T, C)
        x = x.permute(0, 3, 1, 2)       # (B, C, F, T)
        
        return x 