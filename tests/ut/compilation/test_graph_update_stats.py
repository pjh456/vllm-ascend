# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the UpdatableGraph host-side cost accounting.

The stats object is pure Python, so these run without an NPU: they only pin the
aggregation and the reset-on-report behaviour that the profiling report relies
on.
"""

from unittest.mock import patch

import pytest

from vllm_ascend.compilation.updatable_graph import GraphUpdateStats


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_GRAPH_UPDATE_PROFILE", raising=False)
    assert GraphUpdateStats().enabled is False


def test_enabled_through_env(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_GRAPH_UPDATE_PROFILE", "1")
    assert GraphUpdateStats().enabled is True


def test_add_step_accumulates():
    stats = GraphUpdateStats(enabled=True)
    stats.add_step(resolve_s=0.001, update_s=0.005)
    stats.add_step(resolve_s=0.003, update_s=0.007)

    assert stats.steps == 2
    assert stats.resolve_s == pytest.approx(0.004)
    assert stats.update_s == pytest.approx(0.012)


def test_report_logs_once_per_window_and_resets():
    stats = GraphUpdateStats(enabled=True)
    stats.LOG_EVERY_STEPS = 2

    with patch("vllm_ascend.compilation.updatable_graph.logger") as fake_logger:
        stats.add_step(0.001, 0.005)
        stats.add_step(0.001, 0.005)
        assert fake_logger.info.call_count == 1
        # Report resets the window.
        assert stats.steps == 0
        assert stats.update_s == 0.0

        stats.add_step(0.001, 0.005)
        assert fake_logger.info.call_count == 1


def test_report_without_samples_is_noop():
    stats = GraphUpdateStats(enabled=True)
    with patch("vllm_ascend.compilation.updatable_graph.logger") as fake_logger:
        stats.report()
        fake_logger.info.assert_not_called()


def test_disabled_does_not_log_or_reset():
    stats = GraphUpdateStats(enabled=False)
    stats.LOG_EVERY_STEPS = 1

    with patch("vllm_ascend.compilation.updatable_graph.logger") as fake_logger:
        stats.add_step(0.001, 0.005)
        fake_logger.info.assert_not_called()
        assert stats.steps == 1


def test_reporting_switches_itself_off_after_max_reports():
    stats = GraphUpdateStats(enabled=True)
    stats.LOG_EVERY_STEPS = 1
    stats.MAX_REPORTS = 2

    with patch("vllm_ascend.compilation.updatable_graph.logger") as fake_logger:
        for _ in range(2):
            stats.add_step(0.001, 0.005)
        assert fake_logger.info.call_count == 2
        assert stats.enabled is False

        # No further accounting once the report budget is exhausted.
        stats.add_step(0.001, 0.005)
        assert fake_logger.info.call_count == 2


def test_phase_totals_are_reported_separately():
    stats = GraphUpdateStats(enabled=True)
    stats.tasks = 16
    stats.begin_s = 0.00016
    stats.op_s = 0.00032
    stats.end_s = 0.00064
    stats.record_s = 0.00008
    stats.update_s = stats.begin_s + stats.op_s + stats.end_s + stats.record_s
    stats.steps = 2

    with patch("vllm_ascend.compilation.updatable_graph.logger") as fake_logger:
        stats.report()
        args = fake_logger.info.call_args[0]
        rendered = args[0] % args[1:]
    # 8.0 tasks/step, update 0.6 ms/step, end phase 0.32 ms/step.
    assert "tasks/step=8.0" in rendered
    assert "update 0.600 ms/step" in rendered
    assert "end 0.320" in rendered
