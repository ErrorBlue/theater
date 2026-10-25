# D6：采样器与 generate 循环

> 目标：把 logits 变成下一个 token，并写一个能停、能复现、不会越界的生成循环。

## D6 完成标准

- [ ] `temperature=0` 时输出 argmax，且结果确定。
- [ ] top-k 只保留 k 个候选。
- [ ] top-p 至少保留 1 个候选，不产生全 `-inf`。
- [ ] 固定 seed 下生成可复现。
- [ ] 生成遇到 EOS 或达到 `max_new_tokens` 能停止。
- [ ] `pytest -q tests/test_sampler.py` 通过。

## 1. 采样器

```python
def sample(logits, temperature, top_k, top_p, generator):
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum - probs >= top_p   # 保留第一个超过 p 的 token
        remove[..., 0] = False           # 至少保留概率最高的 1 个
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
```

要点：

- `temperature=0` 直接 argmax；不要除以 0。
- top-k 的 `k` 不能超过词表大小。
- top-p 至少要保留 1 个 token。
- 采样前 `model.eval()`，生成入口用 `torch.inference_mode()`。
- 固定 seed + 固定设备，生成应可复现。
- 输入 `logits` 是 `[B,V]`，输出是 `[B]`。

## 2. generate 循环

```python
@torch.inference_mode()
def generate(model, tokenizer, prompt, cfg):
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]
    ids = torch.tensor([prompt_ids])   # [1, T]

    cache = KVCache(cfg.model, batch_size=1, dtype=next(model.parameters()).dtype)
    logits = model(ids, cache)
    cache.advance(ids.shape[1])

    generated = []
    eos_id = tokenizer.eos_id
    generator = torch.Generator().manual_seed(cfg.inference.seed)

    for _ in range(cfg.inference.max_new_tokens):
        # 防止 cache 越界：至少给新 token 留 1 个位置
        if cache.length >= cfg.model.max_seq_len:
            break

        next_id = sample(
            logits[:, -1],
            cfg.inference.temperature,
            cfg.inference.top_k,
            cfg.inference.top_p,
            generator,
        )
        generated.append(next_id.item())

        if next_id.item() == eos_id:
            break

        logits = model(next_id[:, None], cache)   # [1,1,V]
        cache.advance(1)

    return tokenizer.decode(generated)
```

要点：

- `prompt_ids` 为空时至少加一个 BOS，避免 `T=0`。
- `max_new_tokens` 只计新生成的 token，不含 prompt。
- `cache.length >= max_seq_len` 时停止，避免写入越界。
- EOS 本身可以加入 `generated`，decode 时会过滤。
- `generator` 在 `generate` 内创建一次并固定 seed；不要每步重置。

## 3. 测试

### greedy 确定

```python
def test_greedy_deterministic():
    logits = torch.randn(2, 10)
    a = sample(logits, temperature=0, top_k=0, top_p=1.0, generator=None)
    b = sample(logits, temperature=0, top_k=0, top_p=1.0, generator=None)
    assert torch.equal(a, b)
    assert torch.equal(a, logits.argmax(-1))
```

### top-k 边界

```python
def test_top_k():
    logits = torch.randn(1, 10)
    out = sample(logits, temperature=1.0, top_k=1, top_p=1.0, generator=None)
    assert out.item() == logits.argmax(-1).item()
```

### top-p 不产生 NaN

```python
def test_top_p_no_nan():
    logits = torch.randn(1, 10)
    out = sample(logits, temperature=1.0, top_k=0, top_p=0.0, generator=None)
    assert torch.isfinite(out).all()
```

### 固定 seed 可复现

```python
def test_sampling_seed():
    logits = torch.randn(1, 10)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = sample(logits, 1.0, 5, 0.9, g1)
    b = sample(logits, 1.0, 5, 0.9, g2)
    assert torch.equal(a, b)
```

## 4. 常见坑

| 坑 | 现象 | 排查 |
|---|---|---|
| `temperature=0` 还做除法 | Inf/NaN | 直接 argmax |
| top-p 全 mask | softmax NaN | 至少保留 1 个 |
| top-k 大于词表 | `topk` 报错 | `k = min(k, V)` |
| 每步都重置 generator | 采样不可复现 | 只创建一次 |
| 忘记 `model.eval()` | 结果不稳定 | 生成入口先 eval |
| 忘记 `inference_mode` | 内存占用升高 | 装饰器或上下文 |
| decode 输入形状错 | `[1]` vs `[1,1]` | `next_id[:, None]` |
| 超过 `max_seq_len` | cache 越界 | 生成前检查长度 |

## D6 结束前

- [ ] greedy 确定。
- [ ] top-k/top-p 边界正常。
- [ ] 固定 seed 可复现。
- [ ] 生成能因 EOS 或长度停止。
- [ ] `git add -A && git commit -m "d6: sampler + generate loop"`。

下一步：[D7_cli_checkpoint.md](D7_cli_checkpoint.md)。
