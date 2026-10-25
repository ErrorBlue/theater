# v0.1.0 实施计划

> 项目：`theater`  
> 目标版本：v0.1.0  
> 原则：**只做模型架构 + 推理链路；只在 CPU 上运行；不训练、不接数据、不做推理优化。**

## 当前状态

项目使用 `venv` + `pip` 管理：

```text
theater/
├── pyproject.toml          # name = "theater"
├── requirements.txt        # 运行时 + 开发依赖
├── README.md
├── .gitignore
├── docs/
└── theater/
    └── __init__.py         # 示例 main()
```

依赖统一写在 `requirements.txt`：`torch`、`numpy`、`pyyaml`、`tqdm`、`pytest`、`ruff`。

## 最终目标

完成这条链路，并且每一段都能单独测试：

```text
Prompt 文本
  → ByteTokenizer.encode
  → Token IDs
  → Embedding
  → N × Transformer Block
  → Final RMSNorm
  → LM Head
  → Logits
  → Sampler
  → Next Token
  → KV Cache 追加
  → 循环生成
  → ByteTokenizer.decode
  → 输出文本
```

**v0.1.0 不训练。** 权重是随机初始化的，生成结果是乱码，这是正常现象。本版验收的是结构、形状、因果性、KV Cache 一致性和生成流程，不是生成质量。

## 8 天计划

每天只打开当天文档，做完当天的检查表再进入下一天。

| 天 | 文档 | 任务 | 当天产出 |
|---|---|---|---|
| D1 | [D1_environment.md](D1_environment.md) | venv 依赖、目录、配置、ByteTokenizer、空模型 forward | 项目能跑、文本能编码、logits 形状正确 |
| D2 | [D2_norm_rope.md](D2_norm_rope.md) | RMSNorm、RoPE | 两个模块通过数值/相对位置测试 |
| D3 | [D3_attention.md](D3_attention.md) | 因果多头注意力、mask | attention 形状与因果性测试通过 |
| D4 | [D4_transformer.md](D4_transformer.md) | GQA、SwiGLU、Block、完整模型、初始化 | 完整模型输出 `[B,T,V]`，参数量对账 |
| D5 | [D5_kv_cache.md](D5_kv_cache.md) | KV Cache、prefill/decode | 增量 decode 与全量 forward 一致 |
| D6 | [D6_sampling_generate.md](D6_sampling_generate.md) | 采样器、generate 循环、EOS | 贪心确定、采样不越界、生成能停 |
| D7 | [D7_cli_checkpoint.md](D7_cli_checkpoint.md) | CLI、checkpoint、benchmark | 三个子命令可跑，checkpoint 可复现 |
| D8 | [D8_testing.md](D8_testing.md) | 全量测试、演示、收尾 | `pytest` 全绿，打 `v0.1.0` tag |

## 最终验收

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 环境 | `pip install -r requirements.txt` 成功；`python -c "import torch"` 成功 |
| 2 | 前向 | 输入 `[B,T]`，输出 logits `[B,T,V]`，无 NaN |
| 3 | 因果性 | 修改位置 `t` 之后的 token，不改变位置 `≤t` 的 logits |
| 4 | KV Cache | 增量 decode 与全量 forward 的 logits 最大绝对误差 < 1e-5（fp32） |
| 5 | RoPE | 相对位置性质测试通过；不同 offset 下结果正确 |
| 6 | GQA | `n_heads % n_kv_heads == 0`；形状与 repeat 逻辑测试通过 |
| 7 | 采样 | `temperature=0` 时 greedy 确定；top-k/top-p 不越界、不产生 NaN |
| 8 | 生成 | CLI 能完成指定 `max_new_tokens`，遇到 EOS 停止 |
| 9 | Checkpoint | 保存后重新加载，固定种子下生成结果一致 |
| 10 | 测试与文档 | `pytest -q` 全绿；README 有运行命令和已知限制 |

## 前置要求

| 领域 | 需要达到的程度 |
|---|---|
| Python | 函数、类、模块、类型注解、异常、文件读写、命令行 |
| PyTorch | Tensor 运算、广播、`nn.Module`、autograd、`optim` |
| 数学 | 矩阵乘法、转置、softmax、交叉熵、链式法则 |
| 深度学习 | MLP、激活函数、归一化、Transformer 基本概念 |
| 工程 | git、命令行、YAML/JSON、pytest |

如果 PyTorch 还不熟，先花 1–2 周看 Karpathy 的 `micrograd`、`makemore` 和 “Let's build GPT”，再回来做本版。不要一边学 `nn.Module` 一边写 KV Cache。

## 每天怎么用

1. 打开当天的 `Dn_*.md`。
2. 按顺序实现，不提前做后面文档的内容。
3. 跑当天文档里的测试命令。
4. 勾完“当天完成标准”。
5. `git add -A && git commit -m "dN: ..."`。
6. 再进入下一天。

## 现在开始

```powershell
# D1：创建虚拟环境并安装依赖
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# 验证
python -c "import torch; print(torch.__version__)"
```

然后打开 [D1_environment.md](D1_environment.md)。

## 只读参考

- [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [PyTorch 官方基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

先自己写，卡住超过 45 分钟再看参考；不要复制整段实现。
