# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 checkpoints with an untied lm_head load and use it (#5170)."""

import pytest

pytest.importorskip("mlx_vlm.models.gemma4")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_vlm.models.gemma4 import gemma4  # noqa: E402
from mlx_vlm.models.gemma4.config import (  # noqa: E402
    ModelConfig,
    TextConfig,
    VisionConfig,
)

from omlx.patches.gemma4_untied_lm_head import (  # noqa: E402
    apply_gemma4_untied_lm_head_patch,
)

HIDDEN = 64
VOCAB = 512


@pytest.fixture(scope="module", autouse=True)
def _patched():
    apply_gemma4_untied_lm_head_patch()


def _model(untied: bool) -> gemma4.Model:
    text = TextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=5,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=0,
        sliding_window=16,
        tie_word_embeddings=not untied,
    )
    config = ModelConfig(
        text_config=text,
        vision_config=VisionConfig(),
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        tie_word_embeddings=not untied,
    )
    return gemma4.Model(config)


def test_untied_checkpoint_loads_and_logits_use_lm_head():
    model = _model(untied=True)
    weights = dict(tree_flatten(model.parameters()))
    weights["lm_head.weight"] = mx.random.normal((VOCAB, HIDDEN))
    weights = model.sanitize(weights)

    model.load_weights(list(weights.items()), strict=True)

    hidden = mx.random.normal((1, 3, HIDDEN))
    cap = model.language_model.final_logit_softcapping
    expected = mx.tanh((hidden @ weights["lm_head.weight"].T) / cap) * cap
    assert mx.allclose(
        model.language_model.logits_from_hidden(hidden), expected, atol=1e-4
    )


def test_untied_lm_head_quantizes_with_the_checkpoint_keys():
    model = _model(untied=True)
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        class_predicate=lambda path, m: (
            path == "lm_head" and hasattr(m, "to_quantized")
        ),
    )
    keys = dict(tree_flatten(model.parameters()))
    assert {"lm_head.weight", "lm_head.scales", "lm_head.biases"} <= keys.keys()


def test_tied_model_has_no_lm_head_and_ignores_it():
    model = _model(untied=False)
    assert not any("lm_head" in k for k, _ in tree_flatten(model.parameters()))
    hidden = mx.random.normal((1, 2, HIDDEN))
    embed = model.language_model.model.embed_tokens
    cap = model.language_model.final_logit_softcapping
    expected = mx.tanh(embed.as_linear(hidden) / cap) * cap
    assert mx.allclose(
        model.language_model.logits_from_hidden(hidden), expected, atol=1e-4
    )


async def test_engine_start_applies_the_patch(tmp_path, monkeypatch):
    import mlx_vlm.utils as vlm_utils
    from mlx_vlm.models.gemma4 import language

    from omlx.engine import vlm as vlm_module
    from omlx.patches import gemma4_untied_lm_head
    from omlx.utils import model_loading

    class LoadReachedError(Exception):
        pass

    def unpatched_init(self, config, *args, **kwargs):
        raise AssertionError("never called")

    def unpatched_logits(self, hidden):
        raise AssertionError("never called")

    def stop_at_load(model_path, **kwargs):
        raise LoadReachedError

    monkeypatch.setattr(gemma4_untied_lm_head, "_PATCHED", False)
    monkeypatch.setattr(gemma4.Model, "__init__", unpatched_init)
    monkeypatch.setattr(language.LanguageModel, "logits_from_hidden", unpatched_logits)
    monkeypatch.setattr(vlm_utils, "load", stop_at_load)
    monkeypatch.setattr(
        model_loading, "maybe_apply_pre_load_patches", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        model_loading, "maybe_load_custom_quantization", lambda *a, **kw: None
    )
    (tmp_path / "config.json").write_text('{"model_type": "gemma4"}')

    engine = vlm_module.VLMBatchedEngine(model_name=str(tmp_path))
    with pytest.raises(LoadReachedError):
        await engine.start()

    assert gemma4.Model.__init__ is not unpatched_init
    assert language.LanguageModel.logits_from_hidden is not unpatched_logits
