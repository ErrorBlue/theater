from dataclasses import dataclass
from pathlib import Path

import yaml


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

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")


@dataclass
class InferenceConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    seed: int = 42

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")


@dataclass
class Config:
    model: ModelConfig
    inference: InferenceConfig


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(
        model=ModelConfig(**raw["model"]),
        inference=InferenceConfig(**raw["inference"]),
    )
    return cfg
