"""Public motor-boundary and maintenance contracts used by Joint Host."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Event

import numpy as np
import pytest


def test_coherent_motor_feedback_is_aligned_and_immutable(connected_mock_hand):
    hand = connected_mock_hand
    ids = tuple(hand.config.motor_ids)
    targets = np.linspace(0.1, 0.5, len(ids))
    hand.write_motor_pos_strict(ids, targets)
    feedback = hand.read_motor_feedback()
    assert feedback.motor_ids == ids
    assert feedback.positions_rad == pytest.approx(targets)
    assert len(feedback.velocities_rad_s) == len(ids)
    assert len(feedback.currents_ma) == len(ids)
    assert feedback.read_ok == (True,) * len(ids)
    with pytest.raises(FrozenInstanceError):
        feedback.positions_rad = ()
    hand.write_motor_pos_strict(ids[:1], [0.7])
    assert feedback.positions_rad == pytest.approx(targets), "snapshot must not alias motor storage"


@pytest.mark.parametrize("indices,positions", [
    ([], []), ([0, 0], [0.1, 0.2]), ([0], [float("nan")]),
    ([0], [float("inf")]), ([0], [0.1, 0.2]), ([0], [[0.1]]),
])
def test_strict_write_rejects_malformed_requests_before_transport(
    connected_mock_hand, monkeypatch, indices, positions
):
    hand = connected_mock_hand
    calls = []
    monkeypatch.setattr(hand.motor_client, "write_desired_pos", lambda *a, **kw: calls.append(a))
    ids = [hand.config.motor_ids[i] for i in indices]
    with pytest.raises(ValueError):
        hand.write_motor_pos_strict(ids, positions)
    assert calls == []


def test_strict_write_rejects_foreign_motor_ids(connected_mock_hand):
    hand = connected_mock_hand
    with pytest.raises(ValueError, match="configured motor IDs"):
        hand.write_motor_pos_strict([max(hand.config.motor_ids) + 1], [0.1])


@pytest.mark.parametrize("operation", ["calibrate", "tension"])
def test_precancelled_maintenance_never_enables_and_reports_cleanup_failure(
    connected_mock_hand, monkeypatch, operation
):
    hand = connected_mock_hand
    cancelled = Event()
    cancelled.set()
    enables = []
    monkeypatch.setattr(hand, "enable_torque", lambda *a, **kw: enables.append(a))
    failed_motor = hand.config.motor_ids[0]
    monkeypatch.setattr(hand, "disable_torque", lambda: [failed_motor])
    outcome = getattr(hand, operation)(blocking=True, cancel_event=cancelled)
    assert outcome.cancelled and not outcome.completed
    assert outcome.cleanup_torque_disable_failed_ids == (failed_motor,)
    assert enables == []
    assert cancelled.is_set()
    assert hand.last_maintenance_outcome == outcome
    assert not hand.task_running


def test_stop_waits_for_blocking_task_not_executor_thread(connected_mock_hand):
    hand = connected_mock_hand
    entered = Event()
    cancelled = Event()

    def progress(event):
        if event.get("phase") == "holding":
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            hand.tension, move_motors=False, blocking=True,
            cancel_event=cancelled, progress_callback=progress,
        )
        assert entered.wait(2)
        assert hand.task_running
        try:
            assert hand.stop_task(timeout=2), "the executor worker remains alive between tasks"
        finally:
            cancelled.set()
        outcome = future.result(timeout=2)
        assert outcome.cancelled
        assert outcome.cleanup_torque_disable_failed_ids == ()
        assert outcome.cleanup_error is None
        assert executor.submit(lambda: 7).result(timeout=2) == 7


def test_cancelling_during_winding_never_reenables(connected_mock_hand, monkeypatch):
    hand = connected_mock_hand
    cancelled = Event()
    enables_after_cancel = []
    original_enable = hand.enable_torque

    def enable(*args, **kwargs):
        if cancelled.is_set():
            enables_after_cancel.append(True)
        return original_enable(*args, **kwargs)

    def progress(event):
        if event.get("event") == "winding_progress":
            cancelled.set()

    monkeypatch.setattr(hand, "enable_torque", enable)
    outcome = hand.tension(cancel_event=cancelled, progress_callback=progress)
    assert outcome.cancelled
    assert enables_after_cancel == []
    assert not any(hand.motor_client._torque_enabled.values())


def test_cleanup_restoration_error_does_not_skip_disable(connected_mock_hand, monkeypatch):
    hand = connected_mock_hand
    cancelled = Event()
    cancelled.set()
    disabled = []

    def fail_restore(*args):
        raise OSError("limit write refused")

    monkeypatch.setattr(hand, "set_max_current", fail_restore)
    monkeypatch.setattr(hand, "disable_torque", lambda: disabled.append(True) or [])
    outcome = hand.tension(cancel_event=cancelled)
    assert disabled == [True]
    assert outcome.cancelled
    assert "limit write refused" in outcome.cleanup_error


def test_denied_public_write_preserves_motor_state_and_supports_reentrant_access(connected_mock_hand):
    hand = connected_mock_hand
    before = hand.read_motor_feedback().positions_rad
    with hand.motor_access():
        with pytest.raises(PermissionError):
            hand.write_motor_pos_strict(hand.config.motor_ids, [0.4] * len(before), permitted=lambda: False)
        assert hand.read_motor_feedback().positions_rad == before


def test_feedback_health_probe_preserves_no_fallback_request(connected_mock_hand, monkeypatch):
    hand = connected_mock_hand
    client = hand.motor_client
    read = client.read_position_velocity_current
    requests = []

    def capture(*, allow_fallback=True):
        requests.append(allow_fallback)
        return read(allow_fallback=allow_fallback)

    monkeypatch.setattr(client, "read_position_velocity_current", capture)
    assert all(hand.read_motor_feedback(allow_fallback=False).read_ok)
    assert all(hand.read_motor_feedback().read_ok)
    assert requests == [False, True]
