# D1：环境、项目骨架与 ByteTokenizer

> 目标：用 venv 建好环境、装好依赖，建立 v0.1.0 目录，写出配置、ByteTokenizer 和一个只有 embedding 的空模型，验证 `[B,T] -> [B,T,V]`。

## D1 完成标准

- [ ] `.venv` 创建成功，`pip install -r requirements.txt` 与 `pip install -e .` 成功，`torch`、`numpy`、`pyyaml`、`tqdm`、`pytest`、`ruff` 已安装。
- [ ] `configs/debug.yaml` 能加载并通过校验。
- [ ] `ByteTokenizer` 对英文、中文、emoji 都能 round-trip。
- [ ] 空模型 forward 输出 `[B,T,V]`，无 NaN。
- [ ] `pytest -q tests/test_tokenizer.py` 通过。

## 1. 安装依赖

v0.1.0 只在 CPU 上运行，按默认方式安装 PyTorch 即可。依赖统一写在 `requirements.txt`。在项目根目录执行：

```powershell
# 创建并激活虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 安装项目依赖（含 pytest、ruff）
pip install -r requirements.txt
# 以可编辑模式安装本项目，使 `import theater` 可用
pip install -e .
```

验证：

```powershell
python -c "import torch; print(torch.__version__)"
```

> 如果希望明确使用 CPU-only wheel，可以先单独安装：
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`，再执行 `pip install -r requirements.txt`。

## 2. 建立目录

D1 只创建这些：

```text
theater/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .gitignore
├── docs/
├── configs/
│   ├── debug.yaml
│   └── tiny.yaml
├── theater/
│   ├── __init__.py          # 示例 main()，可保留
│   ├── config.py
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── byte.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── transformer.py   # D1 先放 embedding-only 空壳
│   └── utils/
│       ├── __init__.py
│       └── seed.py
└── tests/
    ├── test_config.py
    └── test_tokenizer.py
```

`inference/`、`training/`、`data/` 这些目录 D1 不创建。

## 3. 配置

`configs/debug.yaml`：

```yaml
model:
  d_model: 128
  n_layers: 2
  n_heads: 4
  n_kv_heads: 2
  d_ff: 256
  max_seq_len: 64
  vocab_size: 260
  rope_base: 10000.0
  norm_eps: 1.0e-6
  tie_embeddings: true

inference:
  max_new_tokens: 16
  temperature: 0.0
  top_k: 0
  top_p: 1.0
  seed: 42
```

`configs/tiny.yaml`：

```yaml
model:
  d_model: 384
  n_layers: 6
  n_heads: 6
  n_kv_heads: 2
  d_ff: 1024
  max_seq_len: 512
  vocab_size: 260
  rope_base: 10000.0
  norm_eps: 1.0e-6
  tie_embeddings: true

inference:
  max_new_tokens: 64
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  seed: 42
```

`config.py` 建议用 dataclass：

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    max_seq_len: int
    vocab_size: int
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

@dataclass
class InferenceConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    seed: int = 42

@dataclass
class Config:
    model: ModelConfig
    inference: InferenceConfig
```

启动时校验：

- `d_model % n_heads == 0`
- `n_heads % n_kv_heads == 0`
- `vocab_size == tokenizer.vocab_size`
- `vocab_size >= 260`
- `max_seq_len >= 1`
- `0 <= top_p <= 1`
- `temperature >= 0`

## 4. ByteTokenizer

接口：

```python
class Tokenizer(Protocol):
    vocab_size: int
    eos_id: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
```

实现：

```python
class ByteTokenizer:
    pad_id = 256
    bos_id = 257
    eos_id = 258
    eot_id = 259
    vocab_size = 260

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_ids = [i for i in ids if 0 <= i < 256]
        return bytes(byte_ids).decode("utf-8", errors="replace")
```

要点：

- 每个 byte 映射到 `0..255`，所以不会 OOV。
- 特殊 token 不参与 `bytes()`，decode 前过滤。
- 对多字节 UTF-8 字符，单独 decode 某个 byte 可能产生替换字符；完整序列 decode 必须可逆。
- `encode` 默认加 `<bos>`/`<eos>`；生成 prompt 时可以传 `add_special_tokens=False`，按你的生成策略定。

测试：

```python
@pytest.mark.parametrize("text", ["", "hello", "你好，世界", "emoji: 😀🚀", "a\nb\tc"])
def test_roundtrip(text):
    tok = ByteTokenizer()
    assert tok.decode(tok.encode(text)) == text
```

## 5. 空模型 forward

D1 不写 attention，只验证管道。`model/transformer.py` 先放：

```python
class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)       # [B, T, D]
        logits = self.lm_head(hidden)        # [B, T, V]
        return logits
```

验证：

```python
cfg = load_config("configs/debug.yaml")
model = Transformer(cfg.model)
x = torch.randint(0, cfg.model.vocab_size, (2, 8))
logits = model(x)
assert logits.shape == (2, 8, cfg.model.vocab_size)
assert torch.isfinite(logits).all()
```

D2–D4 会把这个空壳替换成真正的 Transformer。

## 6. 运行检查

```powershell
# 配置
python -c "from theater.config import load_config; print(load_config('configs/debug.yaml'))"

# tokenizer
python -c "from theater.tokenizer.byte import ByteTokenizer; t=ByteTokenizer(); s='你好😀'; print(t.encode(s)); print(t.decode(t.encode(s)))"

# 空模型
python -c "import torch; from theater.config import load_config; from theater.model.transformer import Transformer; c=load_config('configs/debug.yaml'); m=Transformer(c.model); x=torch.randint(0,c.model.vocab_size,(2,8)); print(m(x).shape)"

# 测试
pytest -q tests/test_config.py tests/test_tokenizer.py
```

## 7. 常见问题

| 问题 | 排查 |
|---|---|
| `ModuleNotFoundError: theater` | 先激活 `.venv`，在项目根目录执行 `pip install -e .` |
| YAML 解析失败 | 检查缩进；用 `yaml.safe_load` |
| tokenizer round-trip 失败 | 检查 decode 是否过滤了 256–259 |
| 空模型 logits 有 NaN | 检查输入 id 是否越界；D1 没有 softmax，正常不应 NaN |

## D1 结束前

- [ ] 依赖安装完成，`pyproject.toml` 有依赖。
- [ ] 目录结构建立。
- [ ] `debug.yaml`、`tiny.yaml` 能加载。
- [ ] ByteTokenizer round-trip 测试通过。
- [ ] 空模型 logits 形状正确。
- [ ] `git add -A && git commit -m "d1: project skeleton + tokenizer"`。

下一步：[D2_norm_rope.md](D2_norm_rope.md)。
