# D7：CLI、Checkpoint 与 Benchmark

> 目标：把模型和生成流程包装成命令行工具，支持保存/加载随机权重，并记录推理性能基线。

## D7 完成标准

- [ ] `generate`、`inspect`、`benchmark` 三个子命令可运行。
- [ ] checkpoint 保存后能加载，固定 seed 下生成结果一致。
- [ ] 加载时能发现 config/tokenizer 不匹配。
- [ ] benchmark 输出 prefill/decode tokens/s、TTFT。
- [ ] `pytest -q tests/test_checkpoint.py` 通过。

## 1. CLI 结构

用标准库 `argparse`，入口 `theater/cli.py`：

```text
theater.cli
├── generate   文本 → 生成文本
├── inspect    打印配置、参数量
└── benchmark  prefill/decode 性能
```

命令：

```powershell
python -m theater.cli generate --config configs/debug.yaml --prompt "hello" --max-new-tokens 32 --temperature 0
python -m theater.cli inspect --config configs/tiny.yaml
python -m theater.cli benchmark --config configs/tiny.yaml
```

`generate` 参数：

- `--config`：模型/推理配置。
- `--prompt`：输入文本。
- `--max-new-tokens`：最多生成多少新 token。
- `--temperature`、`--top-k`、`--top-p`、`--seed`：覆盖配置。

`inspect` 输出：

- 参数量（total / trainable）。
- 各模块参数量。
- 配置校验结果。
- dtype（v0.1.0 固定 fp32）。

`benchmark` 输出：

- prefill tokens/s。
- decode tokens/s。
- TTFT（首 token 延迟）。

## 2. 入口点

你初始化时的 `pyproject.toml` 里有：

```toml
[project.scripts]
theater = "theater:main"
```

这会让 `theater` 调用 `theater/__init__.py` 里的示例 `main()`。有两种做法：

1. **先用模块方式**：`python -m theater.cli ...`，不改 `pyproject.toml`。
2. **想要 `theater ...`**：把入口改成：

```toml
[project.scripts]
theater = "theater.cli:main"
```

v0.1 建议先用第 1 种，减少变量。

## 3. Checkpoint

v0.1 不训练，只保存“能重建模型和 tokenizer”的最小信息：

```python
ckpt = {
    "format_version": 1,
    "config": asdict(cfg),
    "model": model.state_dict(),
    "tokenizer": {
        "type": "byte",
        "vocab_size": tokenizer.vocab_size,
        "hash": tokenizer_hash,
    },
    "step": 0,
}
```

保存：

```python
def save_checkpoint(path, ckpt):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
```

加载：

```python
def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["tokenizer"]["vocab_size"] != ckpt["config"]["model"]["vocab_size"]:
        raise ValueError("tokenizer/config vocab_size mismatch")
    return ckpt
```

要点：

- 先写临时文件，再 `os.replace` 原子替换。
- `torch.load` 新版本默认 `weights_only=True`；加载含 config 的 dict 时显式 `weights_only=False`，只加载自己生成的可信文件。
- `model.load_state_dict(..., strict=True)`；缺 key/多 key 都要暴露。
- 固定 seed 后，保存前与加载后生成结果应一致。

## 4. Benchmark

```python
@torch.inference_mode()
def benchmark(model, cfg, prompt_len=32, gen_len=64):
    model.eval()
    ids = torch.randint(0, cfg.model.vocab_size, (1, prompt_len))

    # TTFT + prefill
    t0 = time.perf_counter()
    cache = KVCache(cfg.model, batch_size=1)
    logits = model(ids, cache)
    cache.advance(ids.shape[1])
    t1 = time.perf_counter()

    # decode
    t2 = time.perf_counter()
    for _ in range(gen_len):
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        logits = model(next_id, cache)
        cache.advance(1)
    t3 = time.perf_counter()

    return {
        "ttft_ms": (t1 - t0) * 1000,
        "prefill_tokens_per_s": prompt_len / (t1 - t0),
        "decode_tokens_per_s": gen_len / (t3 - t2),
    }
```

用 `time.perf_counter()` 计时；先 warmup 1–2 次，再取 5 次中位数。v0.1.0 只记录 CPU 数字。

## 5. 测试

```python
def test_checkpoint_roundtrip(tmp_path):
    cfg = load_config("configs/debug.yaml")
    model = Transformer(cfg.model).eval()
    ckpt = {
        "format_version": 1,
        "config": asdict(cfg),
        "model": model.state_dict(),
        "tokenizer": {"type": "byte", "vocab_size": 260, "hash": "test"},
        "step": 0,
    }
    path = tmp_path / "model.pt"
    save_checkpoint(path, ckpt)

    loaded = load_checkpoint(path)
    model2 = Transformer(ModelConfig(**loaded["config"]["model"]))
    model2.load_state_dict(loaded["model"], strict=True)
    model2.eval()

    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    torch.testing.assert_close(model(x), model2(x))
```

再加：

- 保存前/加载后固定 seed 生成结果一致。
- config 与 tokenizer vocab_size 不一致时报错。
- 缺 key/多 key 时报错。
- 临时文件不残留。

## 6. 常见坑

| 坑 | 现象 | 排查 |
|---|---|---|
| checkpoint 只存 state_dict | 加载时不知道 config | 把 config 一起存 |
| tokenizer 版本丢失 | 加载后输出乱码 | 存 vocab_size/hash |
| `torch.load` 默认 weights_only | 加载 dict 报错 | 显式 `weights_only=False` |
| 非原子写入 | 中断后文件损坏 | 临时文件 + `os.replace` |
| benchmark 没 warmup | 数字偏低 | 先跑 1–2 次 |
| `generate` 参数覆盖错 | 配置不生效 | 命令行参数优先 |

## D7 结束前

- [ ] `generate` 能输出文本。
- [ ] `inspect` 参数量正确。
- [ ] `benchmark` 有 TTFT、prefill、decode 数字。
- [ ] checkpoint 保存/加载/严格校验通过。
- [ ] `git add -A && git commit -m "d7: cli + checkpoint + benchmark"`。

下一步：[D8_testing.md](D8_testing.md)。
