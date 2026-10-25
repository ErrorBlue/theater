# D8：全量测试、演示与收尾

> 目标：把 D1–D7 的模块串起来，跑全量测试，完成演示，打 `v0.1.0` tag。

## D8 完成标准

- [ ] `pytest -q` 全绿。
- [ ] 因果性、KV Cache 一致性、checkpoint 一致性三项都有测试。
- [ ] CLI 三个子命令可运行。
- [ ] benchmark 有 CPU 数字。
- [ ] README 写清楚本版做了什么、不做什么、怎么运行。
- [ ] `git tag v0.1.0`。

## 1. 最终验收

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

## 2. 测试矩阵

| 测试文件 | 覆盖内容 | 对应天 |
|---|---|---|
| `test_config.py` | 配置加载、字段校验、vocab_size 一致性 | D1 |
| `test_tokenizer.py` | round-trip、特殊 token、Unicode | D1 |
| `test_norm.py` | 形状、数值、fp32 稳定性 | D2 |
| `test_rope.py` | 相对位置、offset、与朴素实现一致 | D2 |
| `test_attention.py` | 形状、因果性、mask、无 NaN | D3 |
| `test_model.py` | logits 形状、参数量、tie embeddings | D4 |
| `test_cache.py` | prefill/decode 与全量 forward 一致 | D5 |
| `test_sampler.py` | greedy、top-k、top-p、seed | D6 |
| `test_checkpoint.py` | 保存/加载、严格校验、生成一致 | D7 |

## 3. 必须有的测试

### 因果性

```python
def test_causality():
    torch.manual_seed(0)
    cfg = load_config("configs/debug.yaml").model
    model = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % cfg.vocab_size

    y1 = model(x)
    y2 = model(x2)
    torch.testing.assert_close(y1[:, :-1], y2[:, :-1], atol=1e-6, rtol=1e-6)
```

### KV Cache 一致性

```python
def test_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = load_config("configs/debug.yaml").model
    model = Transformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    full = model(ids)

    cache = KVCache(cfg)
    logits = model(ids[:, :1], cache)
    cache.advance(1)
    outs = [logits[:, -1]]
    for t in range(1, ids.shape[1]):
        logits = model(ids[:, t:t+1], cache)
        cache.advance(1)
        outs.append(logits[:, -1])

    inc = torch.stack(outs, dim=1)
    torch.testing.assert_close(inc, full, atol=1e-5, rtol=1e-5)
```

### Checkpoint 一致性

```python
def test_checkpoint_roundtrip(tmp_path):
    cfg = load_config("configs/debug.yaml")
    model = Transformer(cfg.model).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, {
        "format_version": 1,
        "config": asdict(cfg),
        "model": model.state_dict(),
        "tokenizer": {"type": "byte", "vocab_size": 260, "hash": "test"},
        "step": 0,
    })

    ckpt = load_checkpoint(path)
    model2 = Transformer(ModelConfig(**ckpt["config"]["model"]))
    model2.load_state_dict(ckpt["model"], strict=True)
    model2.eval()

    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    torch.testing.assert_close(model(x), model2(x))
```

> 真实测试里还要固定 seed 比较生成结果，而不只是比较一次 forward。

## 4. 调试顺序

不要一上来跑完整生成。按这个顺序逐级排查：

1. **形状**：`[B,T] -> [B,T,V]`，打印每一步 shape。
2. **数值**：固定输入、固定 seed，检查无 NaN。
3. **因果性**：改未来 token，过去 logits 不变。
4. **RoPE**：相对位置测试通过。
5. **GQA**：`H=Hkv` 与 `H=2*Hkv` 都通过。
6. **Cache**：prefill+decode 与全量 forward 一致。
7. **采样**：greedy 确定，top-k/top-p 边界正常。
8. **生成**：CLI 能停、能 decode、能复现。
9. **Checkpoint**：保存/加载一致。
10. **Benchmark**：记录基线。

## 5. 常见坑

| 坑 | 现象 | 排查 |
|---|---|---|
| mask 方向反了 | 模型能看见未来 | 打印 mask，确认上三角为 `False` |
| 缩放因子错 | 注意力过尖/不稳定 | 用 `1/sqrt(head_dim)` |
| RoPE 成对方式错 | 形状正常但结果不对 | 检查 `0::2`/`1::2` 与 `stack` 顺序 |
| decode 时 offset=0 | 每个新 token 都当位置 0 | 传 `cache.length` |
| cache 写入越界 | 长序列崩溃 | 检查 `end <= max_seq_len` |
| GQA `view` 失败 | 维度不连续 | `expand` 后 `reshape` |
| 忘记 `model.eval()` | 训练态影响结果 | 生成入口先 `eval()` |
| 忘记 `torch.inference_mode()` | 内存占用升高 | 生成入口统一包裹 |
| 手动把 dtype 改成 fp16 | softmax 溢出 NaN | v0.1.0 全程用 fp32 |
| tokenizer 特殊 token 混入 | decode 出现 `<bos>` | decode 时过滤特殊 token |
| top-p 全 mask | 全 `-inf`/NaN | 至少保留概率最高的 1 个 |
| checkpoint 配置不一致 | 加载后输出错乱 | 校验 vocab_size 和 config |
| `torch.load` 默认 weights_only | 加载 dict 报错 | 显式 `weights_only=False` |

## 6. 全量测试

```powershell
pytest -q
ruff check .
ruff format --check .
```

如果测试失败，先跑单个文件：

```powershell
pytest -q tests/test_attention.py -vv
pytest -q tests/test_cache.py -vv
```

## 7. 演示

```powershell
# 参数与配置
python -m theater.cli inspect --config configs/tiny.yaml

# 贪心生成（随机权重，输出乱码是正常的）
python -m theater.cli generate --config configs/debug.yaml --prompt "hello" --max-new-tokens 32 --temperature 0

# 采样生成
python -m theater.cli generate --config configs/debug.yaml --prompt "hello" --max-new-tokens 32 --temperature 0.8 --top-k 50 --top-p 0.95 --seed 42

# 性能基线
python -m theater.cli benchmark --config configs/tiny.yaml
```

演示时明确说明：

- 这是随机权重，输出乱码不是 bug。
- v0.1.0 验证的是结构和链路，不是语言能力。
- 因果性、cache 一致性、checkpoint 一致性比生成文本更重要。

## 8. 收尾

- [ ] 删除临时调试代码和未使用的 import。
- [ ] `README.md` 写清楚：
  - 本版做了什么；
  - 本版不做什么；
  - 环境安装、测试、生成、benchmark 命令；
  - 随机权重会输出乱码。
- [ ] `.gitignore` 排除 `.venv/`、`__pycache__/`、`runs/`、`*.pt`。
- [ ] `git add -A && git commit -m "release: v0.1.0"`。
- [ ] `git tag v0.1.0`。
- [ ] 记录 commit hash 和 benchmark 数字。

> 如果只能完成一件事：先保证因果性和 cache 一致性测试通过。它们比生成出一段“看起来像话”的乱码重要得多。
