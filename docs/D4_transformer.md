# D4：GQA、SwiGLU 与完整 Transformer

> 目标：把 D3 的 MHA 扩展成 GQA，加入 SwiGLU、pre-norm Block，组装完整模型，并核对参数量。今天仍然不做 KV Cache。

## D4 完成标准

- [ ] `n_heads % n_kv_heads == 0` 校验生效。
- [ ] GQA 的 `repeat_kv` 形状正确。
- [ ] SwiGLU 输出形状正确。
- [ ] 完整模型 logits 形状 `[B,T,V]`，无 NaN。
- [ ] 参数量与公式一致。
- [ ] `pytest -q tests/test_model.py` 通过。

## 1. GQA 注意力

GQA：query 有 `H` 个头，key/value 只有 `Hkv` 个头，每个 kv 头被 `H/Hkv` 个 q 头共享。

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        # q: H 个头；k/v: Hkv 个头
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)     # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)   # [B,Hkv,T,Dh]
        v = self.v_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)

        if H != Hkv:
            k = repeat_kv(k, H // Hkv)   # [B,H,T,Dh]
            v = repeat_kv(v, H // Hkv)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)          # [B,H,T,T]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)

        out = att @ v                                             # [B,H,T,Dh]
        out = out.transpose(1, 2).reshape(B, T, D)                # [B,T,D]
        return self.o_proj(out)


def repeat_kv(x, n_rep):
    # x: [B,Hkv,T,Dh] -> [B,Hkv*n_rep,T,Dh]
    if n_rep == 1:
        return x
    B, Hkv, T, Dh = x.shape
    x = x[:, :, None, :, :].expand(B, Hkv, n_rep, T, Dh)
    return x.reshape(B, Hkv * n_rep, T, Dh)
```

要点：

- `expand` 后必须 `reshape`，不能直接 `view`。
- q/k/v 的投影输出维度不同，不能再用一个 `qkv` 线性层。
- `o_proj` 输入是 `H*Dh = D`，输出是 `D`。
- 测试 `H=Hkv`（退化为 MHA）和 `H=2*Hkv` 两种情况。

## 2. SwiGLU

公式：

$$
\operatorname{SwiGLU}(x)=W_2\big(\operatorname{SiLU}(W_1x)\odot W_3x\big)
$$

代码里需要 `import torch.nn.functional as F`。

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

`d_ff` 可取 `8/3 * d_model` 并向上取整到 64 的倍数；`debug`/`tiny` 配置里的 256/1024 也可以直接用。

## 3. Block

```python
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

pre-norm：先归一化再进子层，最后残差相加。

## 4. 完整模型

```python
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)          # [B,T,D]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)             # [B,T,V]
```

要点：

- `tie_embeddings=True` 时，`lm_head.weight` 与 `embed.weight` 是同一个 Parameter。
- 没有 bias。
- 没有 dropout。

## 5. 初始化

v0.1.0 不训练，但初始化要固定，才能保证 checkpoint 和生成结果可复现。

```python
def init_weights(model, n_layers):
    for module in model.modules():
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # 残差分支输出投影缩放；属性名按你的实现调整
    scale = 1.0 / math.sqrt(2 * n_layers)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.o_proj.weight.mul_(scale)
            block.mlp.w2.weight.mul_(scale)
```

固定 seed 后再调用。

## 6. 参数量核对

无 bias、tied embedding 时：

$$
P \approx V D + L\left(2D^2 + 2D H_{kv} D_h + 3D D_{ff} + 2D\right) + D
$$

| 配置 | 参数量 |
|---|---:|
| `debug`：D=128, L=2, H=4, Hkv=2, Dff=256, V=260 | 约 0.33M |
| `tiny`：D=384, L=6, H=6, Hkv=2, Dff=1024, V=260 | 约 9.5M |

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

用公式和 `count_parameters` 对账。

## 7. 测试

```python
def test_model_logits_shape():
    cfg = load_config("configs/debug.yaml").model
    model = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(logits).all()

def test_tie_embeddings():
    cfg = load_config("configs/debug.yaml").model
    model = Transformer(cfg)
    assert model.lm_head.weight is model.embed.weight

def test_parameter_count():
    cfg = load_config("configs/debug.yaml").model
    model = Transformer(cfg)
    expected = (
        cfg.vocab_size * cfg.d_model
        + cfg.n_layers * (
            2 * cfg.d_model ** 2
            + 2 * cfg.d_model * cfg.n_kv_heads * (cfg.d_model // cfg.n_heads)
            + 3 * cfg.d_model * cfg.d_ff
            + 2 * cfg.d_model
        )
        + cfg.d_model
    )
    assert count_parameters(model) == expected
```

## 8. 常见坑

| 坑 | 现象 | 排查 |
|---|---|---|
| GQA 维度错 | `view`/`reshape` 失败 | q 用 H，k/v 用 Hkv |
| `expand` 后 `view` | 报错 | 用 `reshape` |
| 忘记 `o_proj` | 输出表达力不足 | 加输出投影 |
| tie 方式错 | 参数量翻倍 | 直接赋同一个 Parameter |
| 残差顺序错 | 训练不稳定（以后） | pre-norm：`x + sublayer(norm(x))` |
| 参数量对不上 | 结构漏写/多写 | 用公式逐项对账 |

## D4 结束前

- [ ] GQA 形状测试通过。
- [ ] SwiGLU 形状测试通过。
- [ ] 完整模型 logits 形状正确。
- [ ] 参数量与公式一致。
- [ ] `git add -A && git commit -m "d4: gqa + swiglu + transformer"`。

下一步：[D5_kv_cache.md](D5_kv_cache.md)。
