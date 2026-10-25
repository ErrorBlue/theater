# D5：KV Cache 与 prefill/decode

> 目标：让模型记住已经算过的 K/V，prefill 一次处理 prompt，decode 每步只算一个新 token，并且结果与全量 forward 一致。

## D5 完成标准

- [ ] 每层有独立的 K/V cache。
- [ ] prefill 写入 prompt 的 K/V。
- [ ] decode 只输入新 token，追加 K/V。
- [ ] RoPE 使用正确的 `offset`。
- [ ] 增量 decode 与全量 forward 的 logits 最大绝对误差 < 1e-5（fp32）。
- [ ] `pytest -q tests/test_cache.py` 通过。

## 1. 设计

```text
k_cache[layer]: [B, Hkv, max_seq_len, Dh]
v_cache[layer]: [B, Hkv, max_seq_len, Dh]
length: 当前已写入的 token 数
```

- 每层一份 cache，不要只做一个全局 cache。
- 预分配 `max_seq_len`，不要每步 `torch.cat`。
- v0.1 只支持 `B=1`；先不要写 batch padding。
- 只实现两种调用：
  - prefill：`T = prompt_len`，`offset = 0`。
  - decode：`T = 1`，`offset = length`。

## 2. KVCache

```python
class KVCache:
    def __init__(self, cfg, batch_size=1, dtype=torch.float32):
        self.length = 0
        self.k = [
            torch.empty(batch_size, cfg.n_kv_heads, cfg.max_seq_len, cfg.d_model // cfg.n_heads,
                        dtype=dtype)
            for _ in range(cfg.n_layers)
        ]
        self.v = [
            torch.empty_like(self.k[i]) for i in range(cfg.n_layers)
        ]

    def append(self, layer_idx, k, v):
        # k/v: [B, Hkv, T, Dh]
        T = k.shape[-2]
        start = self.length
        end = start + T
        if end > self.k[layer_idx].shape[-2]:
            raise ValueError("KV cache overflow")
        self.k[layer_idx][:, :, start:end, :] = k
        self.v[layer_idx][:, :, start:end, :] = v
        return start, end

    def advance(self, T):
        self.length += T
```

> 注意：`append` 每层写自己的位置，`length` 在所有层写完后统一增加。上面为了简洁把 `advance` 单独列出；实际实现要保证每层写入相同的 `start:end`。

## 3. 带 cache 的 attention

在 D4 的 `CausalSelfAttention.forward` 上加参数：

```python
def forward(self, x, kv_cache=None, layer_idx=0):
    B, T, D = x.shape
    H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.head_dim

    q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
    k = self.k_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)
    v = self.v_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)

    # RoPE 的位置从 cache 的当前长度开始
    offset = kv_cache.length if kv_cache is not None else 0
    q = apply_rope(q, self.cos, self.sin, offset)
    k = apply_rope(k, self.cos, self.sin, offset)

    if kv_cache is not None:
        start, end = kv_cache.append(layer_idx, k, v)
        k_all = kv_cache.k[layer_idx][:, :, :end, :]
        v_all = kv_cache.v[layer_idx][:, :, :end, :]
    else:
        k_all, v_all = k, v

    if H != Hkv:
        k_all = repeat_kv(k_all, H // Hkv)
        v_all = repeat_kv(v_all, H // Hkv)

    att = (q @ k_all.transpose(-2, -1)) / math.sqrt(Dh)   # [B,H,T,offset+T]
    att = att.masked_fill(~causal_mask(T, offset), float("-inf"))
    att = torch.softmax(att, dim=-1)

    out = att @ v_all
    out = out.transpose(1, 2).reshape(B, T, D)
    return self.o_proj(out)
```

> `self.cos`/`self.sin` 要在 attention 或 model 初始化时用 `register_buffer` 注册，避免被当成可训练参数。

### mask 的两种情形

- prefill（`offset=0`）：query 位置 `i` 可以看 key 位置 `0..i`，标准下三角。
- decode（`offset=length, T=1`）：唯一 query 可以看全部 `0..offset`，全 `True`。

```python
def causal_mask(T, offset):
    # query 的绝对位置是 offset + i，key 是 0..offset+T-1
    q_pos = torch.arange(offset, offset + T)[:, None]
    k_pos = torch.arange(offset + T)[None, :]
    return k_pos <= q_pos
```

如果只做 v0.1 的 prefill+逐 token decode，这个通用 mask 足够；不要复用固定 `[T,T]` 的三角 mask。

## 4. 模型 forward

```python
def forward(self, input_ids, kv_cache=None):
    x = self.embed(input_ids)
    for i, block in enumerate(self.blocks):
        x = block(x, kv_cache=kv_cache, layer_idx=i)
    x = self.norm(x)
    return self.lm_head(x)
```

调用方负责：

- prefill：创建 `KVCache`，`logits = model(prompt_ids, cache)`，然后 `cache.advance(prompt_len)`。
- decode：`logits = model(next_id[:, None], cache)`，然后 `cache.advance(1)`。
- 每一步只取 `logits[:, -1]`。

> `advance` 的调用位置要和 `append` 配合好：不要让同一批 token 被写两次，也不要漏写。

## 5. 一致性测试

```python
def test_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = load_config("configs/debug.yaml").model
    model = Transformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))

    # 全量 forward
    full_logits = model(ids)

    # prefill + 逐 token decode
    cache = KVCache(cfg)
    logits = model(ids[:, :1], cache)
    cache.advance(1)
    outs = [logits[:, -1]]

    for t in range(1, ids.shape[1]):
        logits = model(ids[:, t:t+1], cache)
        cache.advance(1)
        outs.append(logits[:, -1])

    inc_logits = torch.stack(outs, dim=1)  # [1, T, V]
    torch.testing.assert_close(inc_logits, full_logits, atol=1e-5, rtol=1e-5)
```

如果误差大，按这个顺序查：

1. RoPE 的 `offset` 是否等于 cache 当前长度。
2. mask 的 query/key 绝对位置是否错位。
3. cache 写入的 `start:end` 是否正确。
4. `length` 是否在正确的时机增加。
5. GQA 的 `repeat_kv` 是否作用在完整 cache 上。

## 6. 常见坑

| 坑 | 现象 | 排查 |
|---|---|---|
| decode 时 offset=0 | 每个 token 都当位置 0 | 传 `cache.length` |
| 复用 `[T,T]` mask | decode 结果错 | 用 `[T, offset+T]` 的绝对位置 mask |
| cache 未预分配 | 越生成越慢 | 预分配 `max_seq_len` |
| 每层共用一个 cache | 层间互相污染 | 每层独立 |
| `length` 增加两次 | 位置跳变 | 每步只 `advance` 一次 |
| 写入越界 | 长序列崩溃 | 检查 `end <= max_seq_len` |
| cache dtype 与模型不一致 | 类型报错 | 用模型的 dtype |

## D5 结束前

- [ ] `KVCache` 每层独立。
- [ ] prefill 写入正确。
- [ ] decode 的 RoPE offset 正确。
- [ ] 增量 decode 与全量 forward 误差 < 1e-5。
- [ ] `git add -A && git commit -m "d5: kv cache + prefill/decode"`。

下一步：[D6_sampling_generate.md](D6_sampling_generate.md)。
