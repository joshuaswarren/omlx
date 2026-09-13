# SPDX-License-Identifier: Apache-2.0
"""HTTP contracts for scheduler admission backpressure."""

import asyncio
import http.client
import json
import socket
import threading
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from omlx.engine.base import AdmissionReservation
from omlx.exceptions import SchedulerQueueFullError
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


def _register_handler(app: FastAPI) -> None:
    import omlx.server as server

    app.add_exception_handler(
        SchedulerQueueFullError,
        server.scheduler_queue_full_handler,
    )


def _build_error_app() -> FastAPI:
    app = FastAPI()
    _register_handler(app)

    @app.get("/v1/raise")
    def raise_queue_full():
        raise SchedulerQueueFullError(current_depth=32, max_depth=32)

    @app.get("/health/raise")
    def raise_queue_full_health():
        raise SchedulerQueueFullError(current_depth=33, max_depth=32)

    return app


def _request(request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt=[1, 2, 3],
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        sampling_params=SamplingParams(),
    )


def _scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(max_num_seqs=1, max_waiting_requests=0)
    scheduler.waiting = deque()
    scheduler.running = {}
    scheduler.prefilling = deque()
    scheduler.requests = {}
    scheduler.block_aware_cache = object()
    scheduler._serialize_llama4_requests = False
    scheduler._generation_overflow_recovery_ids = set()
    scheduler._admission_lock = threading.Lock()
    scheduler._admission_request_ids = set()
    return scheduler


class TestQueueFullHandler:
    def test_returns_503_with_retry_after(self):
        with TestClient(_build_error_app()) as client:
            response = client.get("/v1/raise")

        assert response.status_code == 503
        assert response.headers.get("Retry-After") == "1"

    def test_api_route_uses_openai_error_body(self):
        with TestClient(_build_error_app()) as client:
            response = client.get("/v1/raise")

        body = response.json()
        assert "error" in body
        assert "queue full" in body["error"]["message"].lower()
        assert "32/32" in body["error"]["message"]

    def test_non_api_route_uses_plain_detail(self):
        with TestClient(_build_error_app()) as client:
            response = client.get("/health/raise")

        body = response.json()
        assert "detail" in body
        assert "queue full" in body["detail"].lower()

    def test_scheduler_rejection_crosses_openai_api_boundary(self, monkeypatch):
        import omlx.server as server

        scheduler = _scheduler()
        scheduler.add_request(_request("busy"))

        async def reject_when_busy(*args, **kwargs):
            scheduler.add_request(_request("rejected"))

        engine = MagicMock()
        engine.preflight_chat = AsyncMock(side_effect=reject_when_busy)
        engine.count_chat_tokens = MagicMock(return_value=3)

        async def get_engine_for_model(model_id, *, lease=None):
            return engine

        fake_pool = MagicMock()
        fake_pool.get_entry = MagicMock(return_value=None)
        fake_pool.preload_pinned_models = AsyncMock()
        fake_pool.check_ttl_expirations = AsyncMock()
        fake_pool.shutdown = AsyncMock()
        original_overrides = dict(server.app.dependency_overrides)
        original_engine_pool = server._server_state.engine_pool
        try:
            server.app.dependency_overrides[server.verify_api_key] = lambda: True
            server._server_state.engine_pool = fake_pool
            monkeypatch.setattr(server, "get_engine_for_model", get_engine_for_model)
            monkeypatch.setattr(server, "resolve_model_id", lambda name: name)
            monkeypatch.setattr(
                server, "validate_context_window", lambda *args, **kwargs: None
            )
            with TestClient(server.app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/responses",
                    json={
                        "model": "test-model",
                        "input": "Hello",
                        "stream": False,
                    },
                )
        finally:
            server._server_state.engine_pool = original_engine_pool
            server.app.dependency_overrides.clear()
            server.app.dependency_overrides.update(original_overrides)

        assert response.status_code == 503
        assert response.headers.get("Retry-After") == "1"
        assert response.json()["error"]["type"] == "server_error"


def test_stream_background_releases_unconsumed_admission():
    import omlx.server as server

    scheduler = _scheduler()
    reservation_id = scheduler.reserve_request_admission()
    lease = server._LLMEngineLease()
    lease.hold_admission(AdmissionReservation(scheduler, reservation_id))
    body_started = False

    async def body():
        nonlocal body_started
        body_started = True
        yield b"unused"

    response = StreamingResponse(
        body(),
        background=BackgroundTask(lease.release),
    )
    asyncio.run(response.background())

    assert body_started is False
    assert scheduler._admission_request_ids == set()


def test_concurrent_stream_rejects_before_second_response_headers(monkeypatch):
    import omlx.server as server

    scheduler = _scheduler()
    stream_started = threading.Event()
    release_stream = threading.Event()

    async def preflight(*args, **kwargs):
        reservation_id = scheduler.reserve_request_admission(
            kwargs.get("request_id")
        )
        return AdmissionReservation(scheduler, reservation_id)

    engine = MagicMock()
    engine.preflight_chat = AsyncMock(side_effect=preflight)
    engine.count_chat_tokens = MagicMock(return_value=3)

    async def get_engine_for_model(model_id, *, lease=None):
        return engine

    async def blocked_stream(*args, **kwargs):
        request = _request("streaming")
        scheduler.add_request(
            request,
            admission_reservation_id=kwargs["admission_reservation_id"],
        )
        stream_started.set()
        try:
            while not release_stream.is_set():
                await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"
        finally:
            with scheduler._admission_lock:
                scheduler.requests.pop(request.request_id, None)
                try:
                    scheduler.waiting.remove(request)
                except ValueError:
                    pass
                scheduler._admission_request_ids.discard(request.request_id)

    fake_pool = MagicMock()
    fake_pool.get_entry = MagicMock(return_value=None)
    fake_pool.preload_pinned_models = AsyncMock()
    fake_pool.check_ttl_expirations = AsyncMock()
    fake_pool.shutdown = AsyncMock()
    original_overrides = dict(server.app.dependency_overrides)
    original_engine_pool = server._server_state.engine_pool
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.app, log_level="error", lifespan="on")
    )
    thread = threading.Thread(
        target=uvicorn_server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    first_connection = None
    second_connection = None
    try:
        server.app.dependency_overrides[server.verify_api_key] = lambda: True
        server._server_state.engine_pool = fake_pool
        monkeypatch.setattr(server, "get_engine_for_model", get_engine_for_model)
        monkeypatch.setattr(server, "resolve_model_id", lambda name: name)
        monkeypatch.setattr(
            server, "validate_context_window", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(server, "stream_responses_api", blocked_stream)
        thread.start()
        deadline = time.monotonic() + 3
        while not uvicorn_server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert uvicorn_server.started

        headers = {"Content-Type": "application/json"}
        payload = json.dumps(
            {"model": "test-model", "input": "Hello", "stream": True}
        )
        first_connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        first_connection.request("POST", "/v1/responses", payload, headers)
        first_response = first_connection.getresponse()
        assert first_response.status == 200
        assert stream_started.wait(timeout=3)

        second_connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        second_connection.request("POST", "/v1/responses", payload, headers)
        second_response = second_connection.getresponse()
        second_body = json.loads(second_response.read())

        assert second_response.status == 503
        assert second_response.getheader("Retry-After") == "1"
        assert second_body["error"]["type"] == "server_error"
    finally:
        release_stream.set()
        if first_connection is not None:
            first_connection.close()
        if second_connection is not None:
            second_connection.close()
        uvicorn_server.should_exit = True
        if thread.ident is not None:
            thread.join(timeout=3)
        listener.close()
        server._server_state.engine_pool = original_engine_pool
        server.app.dependency_overrides.clear()
        server.app.dependency_overrides.update(original_overrides)

    assert not thread.is_alive()
    assert scheduler._admission_request_ids == set()
