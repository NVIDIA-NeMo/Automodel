from types import SimpleNamespace

from nemo_automodel._transformers.infrastructure import config_has_quantization


def test_none():
    assert config_has_quantization(None) is False


def test_top_level():
    assert config_has_quantization(SimpleNamespace(quantization_config={"format": "pack-quantized"})) is True


def test_nested_text_config():
    # Kimi-K2.5 keeps quantization_config under text_config, not at the top level.
    cfg = SimpleNamespace(text_config=SimpleNamespace(quantization_config={"format": "pack-quantized"}))
    assert config_has_quantization(cfg) is True


def test_absent():
    assert config_has_quantization(SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))) is False
