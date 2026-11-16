"""Tests for the enable_thinking toggle and detect_thinking_default heuristic."""

import json

import pytest

from omlx.model_discovery import (
    detect_preserve_thinking,
    detect_reasoning_effort,
    detect_thinking_default,
)
from omlx.model_settings import ModelSettings

# ---------------------------------------------------------------------------
# detect_thinking_default
# ---------------------------------------------------------------------------


class TestDetectThinkingDefault:
    """Test chat template heuristic for thinking default detection."""

    def test_qwen_pattern_returns_true(self, tmp_path):
        """Qwen3 pattern: thinking is ON by default, only suppressed when
        enable_thinking is explicitly false."""
        template = (
            "{%- if enable_thinking is false -%}\n"
            "  ... suppress thinking ...\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is True

    def test_gemma_default_false_pattern_returns_false(self, tmp_path):
        """Gemma4 pattern: thinking is OFF by default, requires explicit enable."""
        template = "{%- set thinking = enable_thinking | default(false) -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is False

    def test_explicit_default_true_pattern_returns_true(self, tmp_path):
        """Laguna S-2.1 pattern: enable_thinking | default(true) means ON even
        when another flag in the same template defaults to false."""
        template = (
            "{%- set enable_thinking = enable_thinking | default(true) -%}\n"
            "{%- set preserve_thinking = preserve_thinking | default(false) -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is True

    def test_enable_thinking_paren_pattern_returns_false(self, tmp_path):
        """Template that references enable_thinking) returns False."""
        template = "{%- if default(enable_thinking) -%}think{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is False

    def test_no_enable_thinking_returns_none(self, tmp_path):
        """Template without enable_thinking reference returns None."""
        template = "{{ messages[0].content }}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is None

    def test_no_template_files_returns_none(self, tmp_path):
        """Directory without any template file returns None."""
        assert detect_thinking_default(tmp_path) is None

    def test_laguna_template_uses_recommended_serving_default(self, tmp_path):
        """Laguna's effective default follows Poolside's serving recommendation."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "laguna"}))
        (tmp_path / "chat_template.jinja").write_text(
            "{%- set enable_thinking = enable_thinking | default(false) -%}\n"
            "{%- if not enable_thinking -%}</think>{%- else -%}<think>{%- endif -%}"
        )

        assert detect_thinking_default(tmp_path) is True

    def test_laguna_without_thinking_template_returns_none(self, tmp_path):
        """A Laguna config alone is not evidence that its template accepts the flag."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "laguna"}))

        assert detect_thinking_default(tmp_path) is None

    def test_non_laguna_config_without_thinking_template_returns_none(self, tmp_path):
        """Reading config metadata must not enable thinking for other models."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

        assert detect_thinking_default(tmp_path) is None

    def test_jinja_file_takes_priority_over_tokenizer_config(self, tmp_path):
        """chat_template.jinja is preferred over tokenizer_config.json."""
        # Jinja file says Qwen pattern (True)
        (tmp_path / "chat_template.jinja").write_text(
            "{%- if enable_thinking is false -%}suppress{%- endif -%}"
        )
        # tokenizer_config says Gemma pattern (False)
        tc = {"chat_template": "{%- set t = enable_thinking | default(false) -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        assert detect_thinking_default(tmp_path) is True

    def test_falls_back_to_tokenizer_config(self, tmp_path):
        """When no jinja file exists, reads from tokenizer_config.json."""
        tc = {"chat_template": "{%- if enable_thinking is false -%}ok{%- endif -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert detect_thinking_default(tmp_path) is True

    def test_tokenizer_config_without_chat_template_key(self, tmp_path):
        """tokenizer_config.json without chat_template key returns None."""
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_type": "llama"}))
        assert detect_thinking_default(tmp_path) is None

    def test_unrecognized_pattern_returns_none(self, tmp_path):
        """Template with enable_thinking but no recognized pattern returns None."""
        template = "{%- if enable_thinking == 'maybe' -%}hmm{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is None

    def test_malformed_tokenizer_config_returns_none(self, tmp_path):
        """Malformed JSON in tokenizer_config.json returns None gracefully."""
        (tmp_path / "tokenizer_config.json").write_text("not valid json{{{")
        assert detect_thinking_default(tmp_path) is None


# ---------------------------------------------------------------------------
# ModelSettings.enable_thinking field
# ---------------------------------------------------------------------------


class TestModelSettingsEnableThinking:
    """Test enable_thinking field on ModelSettings dataclass."""

    def test_default_is_none(self):
        ms = ModelSettings()
        assert ms.enable_thinking is None

    def test_set_to_true(self):
        ms = ModelSettings(enable_thinking=True)
        assert ms.enable_thinking is True

    def test_set_to_false(self):
        ms = ModelSettings(enable_thinking=False)
        assert ms.enable_thinking is False


# ---------------------------------------------------------------------------
# detect_preserve_thinking
# ---------------------------------------------------------------------------


class TestDetectPreserveThinking:
    """Test chat template heuristic for preserve_thinking support detection."""

    def test_qwen36_pattern_returns_true(self, tmp_path):
        """Qwen 3.6+ pattern: template references preserve_thinking kwarg."""
        template = (
            "{%- if (preserve_thinking is defined and preserve_thinking is true) "
            "or (loop.index0 > ns.last_query_index) -%}\n"
            "  <think>{{ reasoning_content }}</think>\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_preserve_thinking(tmp_path) is True

    def test_no_preserve_thinking_returns_none(self, tmp_path):
        """Template without preserve_thinking reference returns None."""
        template = "{%- if enable_thinking is false -%}suppress{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_preserve_thinking(tmp_path) is None

    def test_no_template_files_returns_none(self, tmp_path):
        """Directory without any template file returns None."""
        assert detect_preserve_thinking(tmp_path) is None

    def test_jinja_file_takes_priority_over_tokenizer_config(self, tmp_path):
        """chat_template.jinja is preferred over tokenizer_config.json."""
        (tmp_path / "chat_template.jinja").write_text(
            "{%- if preserve_thinking -%}keep{%- endif -%}"
        )
        tc = {"chat_template": "{{ messages[0].content }}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        assert detect_preserve_thinking(tmp_path) is True

    def test_falls_back_to_tokenizer_config(self, tmp_path):
        """When no jinja file exists, reads from tokenizer_config.json."""
        tc = {"chat_template": "{%- if preserve_thinking -%}keep{%- endif -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert detect_preserve_thinking(tmp_path) is True

    def test_tokenizer_config_without_chat_template_key(self, tmp_path):
        """tokenizer_config.json without chat_template key returns None."""
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_type": "llama"}))
        assert detect_preserve_thinking(tmp_path) is None

    def test_malformed_tokenizer_config_returns_none(self, tmp_path):
        """Malformed JSON in tokenizer_config.json returns None gracefully."""
        (tmp_path / "tokenizer_config.json").write_text("not valid json{{{")
        assert detect_preserve_thinking(tmp_path) is None


# ---------------------------------------------------------------------------
# ModelSettings.preserve_thinking field
# ---------------------------------------------------------------------------


class TestModelSettingsPreserveThinking:
    """Test preserve_thinking field on ModelSettings dataclass."""

    def test_default_is_none(self):
        ms = ModelSettings()
        assert ms.preserve_thinking is None

    def test_set_to_true(self):
        ms = ModelSettings(preserve_thinking=True)
        assert ms.preserve_thinking is True

    def test_set_to_false(self):
        ms = ModelSettings(preserve_thinking=False)
        assert ms.preserve_thinking is False


@pytest.mark.parametrize(
    "template, levels",
    [
        ("{{ messages }}", None),
        ("Reasoning: {{ reasoning_effort | default('medium') }}", []),
        # Qwen3.8 raises on anything outside its three levels.
        (
            "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
            "{%- elif resolved_reasoning_effort == 'low' %}",
            ["low", "medium", "xhigh"],
        ),
        # GLM-5.3 maps every other value to max; Hunyuan adds no_think.
        (
            "{%- set e = reasoning_effort if reasoning_effort in ['low', 'high'] %}",
            ["low", "high"],
        ),
        (
            "{%- if reasoning_effort not in ['high', 'low', 'no_think'] %}",
            ["low", "high"],
        ),
        # Muse Spark Open checks the keys of a level-to-strength map.
        (
            "{%- set reasoning_effort_map = "
            "{'low': 8, 'medium': 32, 'high': 256, 'xhigh': 512} -%}"
            "{%- if not (reasoning_effort is string and "
            "reasoning_effort in reasoning_effort_map) -%}",
            ["low", "medium", "high", "xhigh"],
        ),
    ],
)
def test_detect_reasoning_effort_reads_the_accepted_levels(tmp_path, template, levels):
    tc = {"chat_template": template}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
    assert detect_reasoning_effort(tmp_path) == levels


# ---------------------------------------------------------------------------
# _merge_chat_template_kwargs pins the discovered thinking default (mf-omlx-03)
# ---------------------------------------------------------------------------


def _patched_server_state(thinking_default):
    """Server state whose pool resolves any model id to an entry with the default."""
    from types import SimpleNamespace

    from omlx.server import ServerState

    entry = SimpleNamespace(thinking_default=thinking_default)
    state = ServerState()
    state.engine_pool = SimpleNamespace(
        resolve_model_id=lambda model_id, settings_manager=None: model_id,
        get_entry=lambda model_id: entry,
    )
    return state


@pytest.mark.parametrize("thinking_default", [True, False])
def test_merge_pins_discovered_thinking_default(thinking_default, monkeypatch):
    """mf-omlx-03: omitted enable_thinking falls back to the detected default."""
    import omlx.server as server_module
    from omlx.api.openai_models import ChatCompletionRequest
    from omlx.server import _merge_chat_template_kwargs

    monkeypatch.setattr(
        server_module, "_server_state", _patched_server_state(thinking_default)
    )

    request = ChatCompletionRequest(model="qwen3", messages=[])
    merged, budget = _merge_chat_template_kwargs(request, ModelSettings())

    assert merged == {"enable_thinking": thinking_default}
    assert budget is None


def test_merge_leaves_unknown_thinking_default_unset(monkeypatch):
    """mf-omlx-03: no detected default keeps the template's internal default."""
    import omlx.server as server_module
    from omlx.api.openai_models import ChatCompletionRequest
    from omlx.server import _merge_chat_template_kwargs

    monkeypatch.setattr(
        server_module, "_server_state", _patched_server_state(None)
    )

    request = ChatCompletionRequest(model="mystery", messages=[])
    merged, _ = _merge_chat_template_kwargs(request, ModelSettings())

    assert merged == {}


def test_merge_request_kwarg_beats_discovered_default(monkeypatch):
    """mf-omlx-03: explicit request enable_thinking still wins over the default."""
    import omlx.server as server_module
    from omlx.api.openai_models import ChatCompletionRequest
    from omlx.server import _merge_chat_template_kwargs

    monkeypatch.setattr(
        server_module, "_server_state", _patched_server_state(True)
    )

    request = ChatCompletionRequest(
        model="qwen3", messages=[], chat_template_kwargs={"enable_thinking": False}
    )
    merged, _ = _merge_chat_template_kwargs(request, ModelSettings())

    assert merged == {"enable_thinking": False}
