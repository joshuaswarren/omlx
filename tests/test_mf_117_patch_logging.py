# SPDX-License-Identifier: Apache-2.0
"""Minefield 117: a compilation-flag kernel patch that fails to apply must log
a WARNING (naming the patch and the stock-path fallback), not a debug line,
while the settings still say the flag is enabled and the engine still loads.
"""

import logging
from types import SimpleNamespace

import pytest

from omlx.engine.batched import BatchedEngine
from omlx.model_settings import ModelSettings


class _Abort(Exception):
    """Unwinds start() after the patch-apply section."""


def _abort_engine_core(*a, **k):
    raise _Abort


async def _start_until_after_patches(monkeypatch):
    from omlx import engine_core
    from omlx.utils import model_loading as ml

    engine = BatchedEngine(
        model_name="test-model", model_settings=ModelSettings()
    )
    monkeypatch.setattr(
        "omlx.engine.batched.get_tokenizer_config", lambda *a, **k: {}
    )
    monkeypatch.setattr(ml, "maybe_apply_pre_load_patches", lambda *a, **k: None)
    monkeypatch.setattr(
        ml, "maybe_load_custom_quantization", lambda *a, **k: (object(), object())
    )
    monkeypatch.setattr(
        ml, "apply_post_load_transforms", lambda model, settings: model
    )
    monkeypatch.setattr(ml, "materialize_lazy_state", lambda model: None)
    monkeypatch.setattr(
        ml, "apply_activation_fp16", lambda model, settings: None
    )
    monkeypatch.setattr("omlx.engine_core.get_mlx_executor", lambda: None)
    monkeypatch.setattr(engine_core, "AsyncEngineCore", _abort_engine_core)
    with pytest.raises(_Abort):
        await engine.start()
    return engine


@pytest.mark.asyncio
async def test_sdpa256_patch_failure_warns(caplog, monkeypatch):
    """mf-117: failing patch apply surfaces as WARNING, engine still loads."""
    import omlx.patches.sdpa256_attention as sdpa256

    def _boom():
        raise RuntimeError("kernel compile failed")

    monkeypatch.setattr(sdpa256, "apply_sdpa256_attention_patch", _boom)
    with caplog.at_level(logging.DEBUG, logger="omlx"):
        await _start_until_after_patches(monkeypatch)

    warning_lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "sdpa256 attention patch" in r.getMessage()
    ]
    assert warning_lines, (
        "expected a WARNING that the sdpa256 patch did not apply; got only "
        + repr([r.levelname for r in caplog.records if "sdpa256" in r.getMessage()])
    )
    assert "stock path" in warning_lines[0]
