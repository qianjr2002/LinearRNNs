import torch
from model_wrapper import RNNWrapper
from ptflops import get_model_complexity_info

def test_rnn_complexity(rnn_type):
    print(f"\n测试 {rnn_type.upper()} 模型复杂度:")
    
    # 创建模型
    model = RNNWrapper(
        rnn_type=rnn_type,
        hidden_size=64,
        freq_size=64
    )
    
    # 测试前向传播
    x = torch.randn(1, 64, 64, 160)  # (B,C,F,T)
    print("输入形状:", x.shape)
    print(f"输入前5个值:\n{x[0, 0, 0, :5]}")
    
    output = model(x)
    print("输出形状:", output.shape)
    print(f"输出前5个值:\n{output[0, 0, 0, :5]}")
    
    # 计算FLOPs和参数量
    input_shape = (64, 64, 160)  # (C, F, T)
    flops, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )
    
    print(f"FLOPs: {flops}")
    print(f"参数量: {params}")
    
    # 打印模型结构
    print("\n模型结构:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数总量: {total_params:,}")

def main():
    # 测试所有RNN类型
    for rnn_type in ['lru', 'slru', 'rwkv']:
        test_rnn_complexity(rnn_type)

if __name__ == "__main__":
    main() 