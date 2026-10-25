# D2：RMSNorm 与 RoPE

> 目标：实现 RMSNorm 和 RoPE，并用数值测试证明它们是对的。今天不写 attention。

## D2 完成标准

- [ ] `RMSNorm` 输出形状与输入一致。
- [ ] `RMSNorm` 与朴素实现数值一致，fp32 无 NaN。
- [ ] `apply_rope` 不改变形状。
- [ ] RoPE 的 q·k 只依赖相对位置差。
- [ ] `offset` 正确：在位置 `offset` 应用旋转，与完整序列对应位置一致。
- [ ] `pytest -q tests/test_norm.py tests/test_rope.py` 通过。

## 1. RMSNorm

公式：

$$
\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}\odot g
$$

实现：

```python
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight
```

要点：

- 在 fp32 中计算均方根，再转回输入 dtype，避免 fp16 下溢。
- `weight` shape `[d]`，初始化为 1。
- 不要 bias。
- 输入 `[B,T,D]`，输出 `[B,T,D]`。

测试：

```python
def test_norm_shape():
    norm = RMSNorm(16)
    x = torch.randn(2, 8, 16)
    assert norm(x).shape == x.shape

def test_norm_matches_naive():
    norm = RMSNorm(16)
    x = torch.randn(2, 8, 16)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
    torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-6)
```

## 2. RoPE

对每个头的维度成对旋转。第 `i` 对的频率：

$$
\theta_i = 10000^{-2i/d_h},\quad d_h=\text{head\_dim}
$$

位置 `p` 的旋转角为 `p * theta_i`。

### 2.1 预计算 cos/sin

```python
def build_rope_cache(max_seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)          # [T, Dh/2]
    return torch.cos(freqs), torch.sin(freqs)
```

### 2.2 应用旋转

```python
def apply_rope(x, cos, sin, offset=0):
    # x: [B, H, T, Dh]
    T = x.shape[-2]
    c = cos[offset:offset + T][None, None, :, :]
    s = sin[offset:offset + T][None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.stack([out1, out2], dim=-1).flatten(-2)
```

要点：

- 只对 q、k 应用，不对 v 应用。
- 维度按 `[x0, x1, x2, x3, ...]` 成对，即 `0::2` 和 `1::2`。
- `cos/sin` shape `[max_seq_len, head_dim/2]`。
- 有 KV Cache 时，decode 的位置不是 0，而是 `past_len`；必须使用 `offset` 切片。
- `offset + T <= max_seq_len`，否则越界。

### 2.3 测试相对位置

RoPE 的关键性质：同一内容在位置 `p` 和 `q` 的 q·k，只依赖 `p-q`。

```python
def test_rope_relative_position():
    B, H, T, Dh = 1, 2, 8, 16
    cos, sin = build_rope_cache(64, Dh)
    q = torch.randn(B, H, 1, Dh)
    k = torch.randn(B, H, 1, Dh)

    def dot(p, qpos):
        qr = apply_rope(q, cos, sin, offset=p)
        kr = apply_rope(k, cos, sin, offset=qpos)
        return (qr * kr).sum()

    # 相同相对距离，点积应相同
    torch.testing.assert_close(dot(0, 3), dot(5, 8), atol=1e-5, rtol=1e-5)
```

### 2.4 测试 offset

```python
def test_rope_offset_matches_full_sequence():
    B, H, T, Dh = 1, 2, 8, 16
    cos, sin = build_rope_cache(64, Dh)
    x = torch.randn(B, H, T, Dh)

    full = apply_rope(x, cos, sin, offset=0)
    for t in range(T):
        one = apply_rope(x[:, :, t:t+1], cos, sin, offset=t)
        torch.testing.assert_close(one, full[:, :, t:t+1], atol=1e-6, rtol=1e-6)
```

## 3. 常见坑

| 坑 | 现象 | 排查 |
|---|---|---|
| 成对方式错 | 形状正常但结果不对 | 检查 `0::2`/`1::2` 与 `stack` 顺序 |
| 对 v 也旋转 | 输出异常 | 只对 q、k 应用 |
| offset 传 0 | decode 每步都当位置 0 | 传 `past_len` |
| cos/sin 长度不够 | 长序列越界 | cache 至少 `max_seq_len` |
| fp16 计算 RMS | 下溢/NaN | 在 fp32 中算均方根 |
| 把 weight 初始化成 0 | 输出恒为 0 | RMSNorm weight 初始化为 1 |

## 4. 实现顺序

1. 写 `model/norm.py`。
2. 写 `tests/test_norm.py`。
3. 写 `model/rope.py`。
4. 写 `tests/test_rope.py`。
5. 跑测试，修到全绿。

## D2 结束前

- [ ] `RMSNorm` 数值测试通过。
- [ ] RoPE 形状测试通过。
- [ ] RoPE 相对位置测试通过。
- [ ] RoPE offset 测试通过。
- [ ] `git add -A && git commit -m "d2: rmsnorm + rope"`。

下一步：[D3_attention.md](D3_attention.md)。
