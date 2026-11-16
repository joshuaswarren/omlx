# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for minefield 78: tool_choice accepted and ignored.

/v1/chat/completions only enforced "auto" (default) and "none"; unsupported
modes ("required", {"type": "required"}, {"type": "function", ...}) were
accepted and silently ignored, so the model could answer in prose instead of
calling the requested tool. The fix rejects those modes with HTTP 400 naming
the mode, before generation.

Mirrors the self-contained mock engine/pool pattern from
tests/integration/test_responses_tool_merge.py.
"""

from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from omlx.engine.base import BaseEngine


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return f"<decoded:{len(tokens)} tokens>"

    def apply_chat_template(self, messages: List[Dict], tokenize: bool = False, **kwargs) -> str:
        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        return "\n".join(parts)


class MockGenerationOutput:
    def __init__(self, text="Chat response.", **kwargs):
        self.text = text
        self.tokens = kwargs.get("tokens", [1, 2, 3])
        self.prompt_tokens = kwargs.get("prompt_tokens", 10)
        self.completion_tokens = kwargs.get("completion_tokens", 5)
        self.finish_reason = kwargs.get("finish_reason", "stop")
        self.tool_calls = kwargs.get("tool_calls", None)
        self.cached_tokens = kwargs.get("cached_tokens", 0)


class RecordingEngine(BaseEngine):
    def __init__(self):
        self._model_name = "test-model"
        self._tokenizer = MockTokenizer()
        self._model_type = "llama"
        self.recorded_chat_kwargs: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type

    @property
    def grammar_compiler(self):
        return None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def generate(self, prompt, **kwargs) -> MockGenerationOutput:
        return MockGenerationOutput()

    async def stream_generate(self, prompt, **kwargs):
        yield MockGenerationOutput()

    def count_chat_tokens(self, messages, tools=None, chat_template_kwargs=None, **kwargs) -> int:
        return 10

    async def chat(self, messages, **kwargs) -> MockGenerationOutput:
        self.recorded_chat_kwargs.append(kwargs)
        return MockGenerationOutput()

    async def stream_chat(self, messages, **kwargs):
        self.recorded_chat_kwargs.append(kwargs)
        yield MockGenerationOutput()

    def get_stats(self) -> Dict[str, Any]:
        return {}

    def get_cache_stats(self):
        return None


class MockEnginePool:
    def __init__(self, engine):
        self._engine = engine
        self._models = [{"id": "test-model", "loaded": True, "pinned": False, "size": 1}]

    @property
    def model_count(self):
        return len(self._models)

    @property
    def loaded_model_count(self):
        return 1

    @property
    def max_model_memory(self):
        return 32 * 1024 * 1024 * 1024

    @property
    def current_model_memory(self):
        return 1

    def get_entry(self, model_id):
        return None

    def resolve_model_id(self, model_id_or_alias, settings_manager=None):
        return model_id_or_alias

    def get_model_ids(self):
        return [m["id"] for m in self._models]

    def get_status(self):
        return {
            "models": self._models,
            "loaded_count": self.loaded_model_count,
            "max_model_memory": self.max_model_memory,
        }

    async def get_engine(self, model_id, **kwargs):
        return self._engine

    async def release_engine(self, model_id):
        pass


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


@pytest.fixture()
def engine():
    return RecordingEngine()


@pytest.fixture()
def client(engine):
    from omlx.server import app, _server_state

    original_pool = _server_state.engine_pool
    original_default = _server_state.default_model
    original_mcp = _server_state.mcp_manager
    original_settings = _server_state.settings_manager

    _server_state.engine_pool = MockEnginePool(engine)
    _server_state.default_model = "test-model"
    _server_state.mcp_manager = None
    _server_state.settings_manager = None

    yield TestClient(app)

    _server_state.engine_pool = original_pool
    _server_state.default_model = original_default
    _server_state.mcp_manager = original_mcp
    _server_state.settings_manager = original_settings


def _post_chat(client, tool_choice):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": TOOLS,
            "tool_choice": tool_choice,
        },
    )


class TestToolChoiceValidation:
    """Minefield 78: unsupported tool_choice modes must fail closed with 400."""

    @pytest.mark.parametrize(
        "mode",
        [
            "required",
            {"type": "required"},
            {"type": "function", "function": {"name": "get_weather"}},
        ],
        ids=["str-required", "dict-required", "named-function"],
    )
    def test_unsupported_mode_rejected_before_generation(self, client, engine, mode):
        resp = _post_chat(client, mode)
        assert resp.status_code == 400
        detail = resp.json()["error"]["message"]
        # Names the rejected mode and states what oMLX supports.
        assert "required" in detail or "function" in detail
        assert "auto" in detail and "none" in detail
        # Fail closed BEFORE generation: the engine must never run.
        assert engine.recorded_chat_kwargs == []

    @pytest.mark.parametrize("mode", ["auto", "none", None], ids=["auto", "none", "unset"])
    def test_supported_modes_still_accepted(self, client, mode):
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": TOOLS,
        }
        if mode is not None:
            body["tool_choice"] = mode
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
