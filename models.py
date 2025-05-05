# models.py
import torch
import torch.nn as nn
from lru import LRU
from slru import SLRU
from rwkv import RWKV

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, zero_mean=True, scale=True, bias=True):
        super().__init__()
        self.zero_mean = zero_mean
        self.scale = scale
        self.eps = eps
        if scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.bias = None

    def forward(self, x):
        if self.zero_mean:
            x = x - x.mean(dim=-1, keepdim=True)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class RNN_alpha(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        intermediate_size,
        dropout_rate=0.1,
        max_position_embeddings=512,
        embedding_size=None,
        rnn_class=LRU  # 将 RNN 类型作为参数传入
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            layer = nn.ModuleDict({
                'rnn': rnn_class(hidden_size=hidden_size, units=hidden_size),
                'rnn_norm': LayerNorm(hidden_size, zero_mean=False, scale=False),
                'ffn': FeedForward(hidden_size, intermediate_size, dropout_rate),
                'ffn_norm': LayerNorm(hidden_size, zero_mean=False, scale=False)
            })
            self.layers.append(layer)

    def get_position_ids(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.dropout(x)

        for layer in self.layers:
            rnn_output = layer['rnn'](hidden_states)
            hidden_states = hidden_states + rnn_output
            hidden_states = layer['rnn_norm'](hidden_states)

            ffn_output = layer['ffn'](hidden_states)
            hidden_states = hidden_states + ffn_output
            hidden_states = layer['ffn_norm'](hidden_states)

        return hidden_states

def create_rnn_alpha_model(
    vocab_size=21128,
    hidden_size=768,
    num_hidden_layers=12,
    intermediate_size=3072,
    dropout_rate=0.1,
    max_position_embeddings=512,
    rnn_class=LRU  # 显式传入 RNN 类型
):
    return RNN_alpha(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        dropout_rate=dropout_rate,
        max_position_embeddings=max_position_embeddings,
        rnn_class=rnn_class
    )

if __name__ == "__main__":
    torch.manual_seed(42)
    input_ids = torch.randint(0, 21128, (1, 160))
    print("输入形状:", input_ids.shape)

    print("\n=== RNN-alpha 输出测试 ===")
    for name, rnn_type in [('LRU', LRU), ('SLRU', SLRU), ('RWKV', RWKV)]:
        model = create_rnn_alpha_model(
            vocab_size=21128,
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=256,
            max_position_embeddings=160,
            rnn_class=rnn_type
        )
        print(f"rnn_type: {rnn_type}")
        outputs = model(input_ids)
        print(f"\n{name}: 输出形状: {outputs.shape} 前5个值:\n{outputs[0, 0, :5]}")
        # 计算模型复杂度
        print("\n=== 模型复杂度分析 ===")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数总量: {total_params:,}")

        # 打印每层参数量
        # print("\n模型结构:")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.numel():,}")

        # 计算推理时间
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            _ = model(input_ids)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            print(f"推理时间: {elapsed_time:.2f} ms\n")
