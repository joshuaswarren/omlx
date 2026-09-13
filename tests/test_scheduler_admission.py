# SPDX-License-Identifier: Apache-2.0
"""Scheduler admission-control contracts."""

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from omlx.exceptions import SchedulerQueueFullError
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


def _scheduler(*, max_num_seqs: int = 1, max_waiting_requests: int | None = 0):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_waiting_requests=max_waiting_requests,
    )
    scheduler.waiting = deque()
    scheduler.running = {}
    scheduler.prefilling = deque()
    scheduler.requests = {}
    scheduler._serialize_llama4_requests = False
    scheduler._generation_overflow_recovery_ids = set()
    scheduler.block_aware_cache = MagicMock()
    scheduler._admission_lock = threading.Lock()
    scheduler._admission_request_ids = set()
    scheduler._cache_freshness_waits = {}
    scheduler._prefix_cache_prepared = set()
    scheduler._throttle_notified_requests = set()
    scheduler._memory_admission_blocked_request_id = None
    scheduler._memory_admission_blocked_since = 0.0
    scheduler._store_cache_admission_blocked_request_id = None
    scheduler._store_cache_admission_blocked_since = 0.0
    scheduler._inflight_store_futures = {}
    scheduler._prefill_states = {}
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.paged_cache_manager = None
    scheduler._cleanup_specprefill = MagicMock()
    scheduler._cleanup_detokenizer = MagicMock()
    scheduler._cleanup_output_parser_session = MagicMock()
    scheduler.model = MagicMock(spec=[])
    scheduler.tokenizer = MagicMock()
    scheduler.tokenizer.encode.return_value = [1, 2, 3]
    scheduler._boundary_cache_snapshots = {}
    scheduler._boundary_snapshot_store = None
    scheduler.finished_req_ids = set()
    scheduler._schedule_deferred_metal_clear = MagicMock()
    scheduler._drop_boundary_snapshots_for_request = MagicMock()
    scheduler._release_paged_cache_for_request = MagicMock()
    scheduler._stream = None
    scheduler._decode_activity_key = "test"
    scheduler._pending_abort_ids = set()
    scheduler._pending_async_removes = {}
    scheduler._inflight_store_info = {}
    scheduler.batch_generator = None
    scheduler._current_sampler_params = None
    scheduler._boundary_snapshot_required = None
    scheduler._cache_rate_tracker = MagicMock()
    scheduler._boundary_snapshot_diagnostics = MagicMock()
    scheduler._last_prefix_cache_lookup = None
    scheduler._request_detokenizers = {}
    scheduler._output_parser_sessions = {}
    scheduler._deferred_clear_at = None
    scheduler._publish_admin_snapshot = MagicMock()
    return scheduler


def _request(request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt=[1, 2, 3],
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        sampling_params=SamplingParams(),
    )


class TestLegacyWaitingQueueCap:
    @pytest.mark.parametrize(("max_num_seqs", "max_waiting"), [(1, 32), (16, 64)])
    def test_unset_option_preserves_legacy_cap(self, max_num_seqs, max_waiting):
        scheduler = _scheduler(
            max_num_seqs=max_num_seqs,
            max_waiting_requests=None,
        )
        for index in range(max_waiting):
            scheduler.waiting.append(_request(f"waiting-{index}"))

        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(_request("over"))

        assert exc.value.current_depth == max_waiting
        assert exc.value.max_depth == max_waiting

    def test_preflight_reservations_obey_legacy_waiting_cap(self):
        scheduler = _scheduler(max_waiting_requests=None)
        for _ in range(32):
            scheduler.reserve_request_admission()

        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.reserve_request_admission()

        assert exc.value.current_depth == 32
        assert exc.value.max_depth == 32

    def test_duplicate_request_precedes_capacity_check(self):
        scheduler = _scheduler(max_waiting_requests=None)
        request = _request("duplicate")
        scheduler.requests[request.request_id] = request
        for index in range(32):
            scheduler.waiting.append(_request(f"waiting-{index}"))

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(request)


class TestConfiguredWaitingQueueCap:
    def test_preflight_reservation_transfers_without_consuming_a_second_slot(self):
        scheduler = _scheduler()
        reservation_id = scheduler.reserve_request_admission()

        scheduler.add_request(
            _request("accepted"), admission_reservation_id=reservation_id
        )

        assert scheduler._admission_request_ids == {"accepted"}
        with pytest.raises(SchedulerQueueFullError):
            scheduler.reserve_request_admission()

    def test_request_id_reservation_transfers_to_same_request_id(self):
        scheduler = _scheduler()
        scheduler.reserve_request_admission("same-id")

        scheduler.add_request(
            _request("same-id"), admission_reservation_id="same-id"
        )

        assert scheduler._admission_request_ids == {"same-id"}
        assert [request.request_id for request in scheduler.waiting] == ["same-id"]

    def test_released_preflight_reservation_frees_capacity(self):
        scheduler = _scheduler()
        reservation_id = scheduler.reserve_request_admission()

        scheduler.release_request_admission(reservation_id)

        replacement = scheduler.reserve_request_admission()
        assert replacement in scheduler._admission_request_ids

    def test_expired_preflight_reservation_is_rejected(self):
        scheduler = _scheduler()

        with pytest.raises(ValueError, match="Invalid or expired admission reservation"):
            scheduler.add_request(
                _request("rejected"), admission_reservation_id="missing"
            )

    def test_zero_reserves_idle_slot_before_scheduler_step(self):
        scheduler = _scheduler()
        scheduler.add_request(_request("first"))

        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(_request("second"))

        assert exc.value.current_depth == 0
        assert exc.value.max_depth == 0
        assert [request.request_id for request in scheduler.waiting] == ["first"]

    def test_concurrent_adds_cannot_claim_the_same_idle_slot(self):
        scheduler = _scheduler()
        barrier = threading.Barrier(2)

        def add(request_id: str):
            barrier.wait(timeout=2)
            try:
                scheduler.add_request(_request(request_id))
            except SchedulerQueueFullError:
                return "rejected"
            return "accepted"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(add, ("one", "two")))

        assert sorted(outcomes) == ["accepted", "rejected"]
        assert len(scheduler.waiting) == 1

    def test_waiting_to_running_transition_does_not_free_capacity(self):
        scheduler = _scheduler()
        first = _request("first")
        scheduler.add_request(first)
        scheduler.waiting.popleft()
        scheduler.running[first.request_id] = first

        with pytest.raises(SchedulerQueueFullError):
            scheduler.add_request(_request("second"))

        scheduler.running.pop(first.request_id)
        scheduler.remove_finished_request(first.request_id)
        scheduler.add_request(_request("replacement"))
        scheduler.remove_finished_request(first.request_id)

        with pytest.raises(SchedulerQueueFullError):
            scheduler.add_request(_request("over"))
        assert [request.request_id for request in scheduler.waiting] == ["replacement"]

    def test_configured_waiting_slots_extend_admission_capacity(self):
        scheduler = _scheduler(max_waiting_requests=2)
        scheduler.add_request(_request("running"))
        scheduler.add_request(_request("waiting-one"))
        scheduler.add_request(_request("waiting-two"))

        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(_request("over"))

        assert exc.value.current_depth == 2
        assert exc.value.max_depth == 2

    def test_failed_preflight_releases_reservation(self):
        scheduler = _scheduler()
        scheduler.block_aware_cache = None
        scheduler.preflight_or_raise = MagicMock(side_effect=RuntimeError("rejected"))

        with pytest.raises(RuntimeError, match="rejected"):
            scheduler.add_request(_request("first"))

        scheduler.block_aware_cache = object()
        scheduler.add_request(_request("second"))
        assert [request.request_id for request in scheduler.waiting] == ["second"]

    def test_prefill_retry_keeps_its_admission_slot(self):
        scheduler = _scheduler()
        scheduler._reclaim_prefill_headroom = MagicMock()
        scheduler._specprefill_active_request_id = None
        request = _request("retrying")
        scheduler.add_request(request)

        scheduler.waiting.clear()
        scheduler.requests.pop(request.request_id)
        scheduler._clear_request_admission_bookkeeping(
            request.request_id, release_admission=False
        )
        requeued = scheduler._requeue_or_fail_prefill(
            request, RuntimeError("Memory limit exceeded")
        )

        assert requeued is True
        with pytest.raises(SchedulerQueueFullError):
            scheduler.add_request(_request("second"))

    def test_each_model_scheduler_has_independent_capacity(self):
        first_model = _scheduler()
        second_model = _scheduler()

        first_model.add_request(_request("first-model"))
        second_model.add_request(_request("second-model"))

        assert len(first_model.waiting) == 1
        assert len(second_model.waiting) == 1


class TestAdmissionPausedField:
    def test_default_false(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._admission_paused = False
        assert scheduler._admission_paused is False


class TestAdmissionTerminalRelease:
    def test_abort_releases_slot(self):
        scheduler = _scheduler()
        scheduler.add_request(_request("aborted"))

        assert scheduler._do_abort_request("aborted") is True

        replacement = scheduler.reserve_request_admission()
        assert scheduler._admission_request_ids == {replacement}

    def test_normal_finish_releases_slot(self):
        scheduler = _scheduler()
        scheduler.add_request(_request("finished"))

        removed = scheduler.remove_finished_request("finished")

        assert removed is not None
        replacement = scheduler.reserve_request_admission()
        assert scheduler._admission_request_ids == {replacement}

    def test_fail_all_releases_slot(self, monkeypatch):
        import omlx.scheduler as scheduler_module

        scheduler = _scheduler()
        scheduler.add_request(_request("failed"))
        monkeypatch.setattr(scheduler_module, "_sync_and_clear_cache", lambda _: None)
        monkeypatch.setattr(
            scheduler_module, "_unregister_uid_rows_for_model", lambda _: None
        )

        assert scheduler.fail_all_requests() == ["failed"]

        replacement = scheduler.reserve_request_admission()
        assert scheduler._admission_request_ids == {replacement}

    def test_generate_batch_partial_rejection_rolls_back_admission(self):
        from omlx.engine_core import EngineCore

        scheduler = _scheduler()
        engine = EngineCore.__new__(EngineCore)
        engine.scheduler = scheduler

        with pytest.raises(SchedulerQueueFullError):
            engine.generate_batch_sync(["first", "rejected"])

        assert scheduler.requests == {}
        assert list(scheduler.waiting) == []
        assert scheduler._admission_request_ids == set()
        replacement = scheduler.reserve_request_admission()
        assert scheduler._admission_request_ids == {replacement}

    def test_reset_releases_untransferred_reservation(self, monkeypatch):
        import omlx.scheduler as scheduler_module

        scheduler = _scheduler()
        scheduler.reserve_request_admission()
        monkeypatch.setattr(
            scheduler_module, "_unregister_uid_rows_for_model", lambda _: None
        )

        scheduler.reset()

        assert scheduler._admission_request_ids == set()
        replacement = scheduler.reserve_request_admission()
        assert scheduler._admission_request_ids == {replacement}
