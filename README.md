# 线性RNN的相关变体
三种快速可并行RNN变体的PyTorch实现：LRU， SLRU和RWKV。

## 简介

- 中文博客：https://kexue.fm/archives/9554
- LRU论文：https://arxiv.org/abs/2303.06349
- RWKV链接：https://github.com/BlinkDL/RWKV-LM

## 并行

线性RNN支持并行算法，可以将O(L)的运算降低到O(log L)，本项目利用的是prefix sum问题的“Upper/Lower算法”来实现RNN并行。

具体细节可以参考中文博客的“[并行化](https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96)”一节


# 复杂度计算
```
python test_complexity.py

测试 LRU 模型复杂度:
输入形状: torch.Size([1, 64, 64, 160])
输入前5个值:
tensor([-0.7547, -0.4243,  0.0925, -2.2227, -1.9195])
输出形状: torch.Size([1, 64, 64, 160])
输出前5个值:
tensor([-0.0864, -0.0925, -0.3033, -0.4365, -0.3046], grad_fn=<SliceBackward0>)
Warning: module LRU is treated as a zero-op.
Warning: module RNNWrapper is treated as a zero-op.
RNNWrapper(
  1.06 M, 98.855% Params, 169.74 MMac, 100.000% MACs, 
  (rnn_layers): ModuleList(
    (0-63): 64 x LRU(
      16.58 k, 1.545% Params, 2.65 MMac, 1.562% MACs, 
      (i_dense): Linear(8.32 k, 0.775% Params, 1.33 MMac, 0.784% MACs, in_features=64, out_features=128, bias=True)
      (o_dense): Linear(8.26 k, 0.769% Params, 1.32 MMac, 0.778% MACs, in_features=128, out_features=64, bias=True)
    )
  )
)
FLOPs: 169.74 MMac
参数量: 1.07 M

模型结构:
可训练参数总量: 1,073,152

测试 SLRU 模型复杂度:
输入形状: torch.Size([1, 64, 64, 160])
输入前5个值:
tensor([ 1.3978,  1.1476,  1.4789, -0.4923,  0.8744])
输出形状: torch.Size([1, 64, 64, 160])
输出前5个值:
tensor([-0.1543, -0.2485, -0.2451, -0.1301, -0.2603], grad_fn=<SliceBackward0>)
Warning: module SLRU is treated as a zero-op.
Warning: module RNNWrapper is treated as a zero-op.
RNNWrapper(
  532.48 k, 98.485% Params, 85.2 MMac, 100.000% MACs, 
  (rnn_layers): ModuleList(
    (0-63): 64 x SLRU(
      8.32 k, 1.539% Params, 1.33 MMac, 1.562% MACs, 
      (i_dense): Linear(4.16 k, 0.769% Params, 665.6 KMac, 0.781% MACs, in_features=64, out_features=64, bias=True)
      (o_dense): Linear(4.16 k, 0.769% Params, 665.6 KMac, 0.781% MACs, in_features=64, out_features=64, bias=True)
    )
  )
)
FLOPs: 85.2 MMac
参数量: 540.67 k

模型结构:
可训练参数总量: 540,672

测试 RWKV 模型复杂度:
输入形状: torch.Size([1, 64, 64, 160])
输入前5个值:
tensor([ 0.6559,  0.5011,  0.7933, -0.1018,  1.6249])
输出形状: torch.Size([1, 64, 64, 160])
输出前5个值:
tensor([0.2944, 0.3329, 0.2139, 0.1851, 0.1807], grad_fn=<SliceBackward0>)
Warning: module RWKV is treated as a zero-op.
Warning: module RNNWrapper is treated as a zero-op.
RNNWrapper(
  1.06 M, 99.237% Params, 170.39 MMac, 100.000% MACs, 
  (rnn_layers): ModuleList(
    (0-63): 64 x RWKV(
      16.64 k, 1.551% Params, 2.66 MMac, 1.562% MACs, 
      (rkv_dense): Linear(12.48 k, 1.163% Params, 2.0 MMac, 1.172% MACs, in_features=64, out_features=192, bias=True)
      (o_dense): Linear(4.16 k, 0.388% Params, 665.6 KMac, 0.391% MACs, in_features=64, out_features=64, bias=True)
    )
  )
)
FLOPs: 170.39 MMac
参数量: 1.07 M

模型结构:
可训练参数总量: 1,073,152
```

## 交流
QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
