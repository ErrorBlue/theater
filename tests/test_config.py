from pathlib import Path

import pytest

from theater.config import Config, InferenceConfig, ModelConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["debug", "tiny"])
def test_load_config_files(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert isinstance(cfg, Config)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.inference, InferenceConfig)
    assert cfg.model.vocab_size == 260
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.model.n_heads % cfg.model.n_kv_heads == 0


def test_missing_model_section_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "inference:\n  max_new_tokens: 4\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        load_config(path)
