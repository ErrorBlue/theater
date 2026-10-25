# D3：因果多头注意力

> 目标：实现标准的多头因果注意力（MHA），先把 mask 和形状写对。今天不做 GQA，D4 再扩展。

## D3 完成标准

- [ ] attention 输入 `[B,T,D]`，输出 `[B,T,D]`。
- [ ] 内部 `q/k/v` 形状为 `[B,H,T,Dh]`。
- [ ] 未来 token 不影响过去位置的输出。
- [ ] mask 上三角为 `False`，softmax 后不产生 NaN。
- [ ] `pytest -q tests/test_attention.py` 通过。

## 1. 结构

```text
x [B,T,D]
  → q_proj / k_proj / v_proj
  → reshape 成 [B,H,T,Dh]
  → q @ k^T / sqrt(Dh)
  → causal mask
  → softmax
  → @ v
  → 合并多头 [B,T,D]
  → o_proj
```

`Dh = D // H`，`D % H == 0`。

## 2. 实现

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)       # [B,H,T,T]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)

        out = att @ v                                          # [B,H,T,Dh]
        out = out.transpose(1, 2).reshape(B, T, D)             # [B,T,D]
        return self.o_proj(out)
```

要点：

- `q/k/v` 的 `view(B,T,H,Dh)` 要求 tensor 连续；如果前面有 `transpose`，先 `.contiguous()`。
- mask 用 `torch.tril`，上三角为 `False`。
- 先 mask 再 softmax；被 mask 的位置是 `-inf`。
- 缩放因子是 `1/sqrt(Dh)`，不是 `1/sqrt(D)`。
- 不要用 `0` 填充被 mask 的位置，否则 softmax 会分配到概率。
- 不要忘记 `o_proj`。

## 3. 测试

### 形状

```python
def test_attention_shape():
    attn = CausalSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 8, 32)
    assert attn(x).shape == x.shape
```

### 因果性

```python
def test_attention_causal():
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(1, 8, 32)
    x2 = x.clone()
    x2[:, -1] += 10.0          # 只改最后一个位置

    y1 = attn(x)
    y2 = attn(x2)
    # 最后一个位置之前的所有输出不应改变
    torch.testing.assert_close(y1[:, :-1], y2[:, :-1], atol=1e-6, rtol=1e-6)
```

### 无 NaN

```python
def test_attention_no_nan():
    attn = CausalSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(2, 8, 32)
    y = attn(x)
    assert torch.isfinite(y).all()
```

## 4. 常见坑

| 坑 | 现象 | 排查 |
|---|---|---|
| mask 方向反了 | 能看见未来 | 打印 mask，确认上三角为 `False` |
| mask 加在 softmax 之后 | 概率不对 | 先 `masked_fill` 再 softmax |
| 缩放因子用 `sqrt(D)` | 注意力过尖/不稳定 | 用 `sqrt(Dh)` |
| `view` 报错 | 维度不连续 | `.contiguous()` 或调整 reshape 顺序 |
| 忘记合并多头 | 输出形状不对 | `[B,H,T,Dh] -> [B,T,D]` |
| 忘记 `o_proj` | 表达力不足 | 加输出投影 |

## 5. 实现顺序

1. 写 `model/attention.py` 的 `CausalSelfAttention`。
2. 写 `tests/test_attention.py` 的形状测试。
3. 加因果性测试。
4. 加无 NaN 测试。
5. 跑测试，修到全绿。

## D3 结束前

- [ ] `CausalSelfAttention` 输出形状正确。
- [ ] 因果性测试通过。
- [ ] 无 NaN。
- [ ] 记录 attention 的参数量。
- [ ] `git add -A && git commit -m "d3: causal multi-head attention"`。

下一步：[D4_transformer.md](D4_transformer.md)。
