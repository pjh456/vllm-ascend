# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the pipelined UpdatableGraph update.

The worker moves the per-step ``graph_task_update_*`` loop off the model-runner
thread. These tests pin the two properties the rest of the runner depends on:

* the barrier keeps the update ordered *after* the model-runner stream without
  touching the update stream once it belongs to the worker thread, and
* a worker failure is surfaced on the model-runner thread instead of silently
  replaying a graph with stale parameters.
"""

import pytest

from vllm_ascend.compilation import updatable_graph as ug


class _FakeStream:
    def __init__(self):
        self.waited = []

    def wait_stream(self, stream):
        self.waited.append(stream)


class _FakeEvent:
    def __init__(self, record=None):
        self.recorded = []
        self.waited = []
        self._record = record

    def record(self, stream=None):
        self.recorded.append(stream)

    def wait(self, stream=None):
        self.waited.append(stream)


class _FakeExecutor:
    def __init__(self):
        self.submitted = []

    def submit(self, graph, update_stream, resolved_tasks, wait_event):
        self.submitted.append((graph, update_stream, resolved_tasks, wait_event))


class _FakeGraph:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def apply_update(self, update_stream, resolved_tasks, wait_event):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((update_stream, resolved_tasks, wait_event))


@pytest.fixture
def barrier_env(monkeypatch):
    monkeypatch.setattr(ug.torch.npu, "current_stream", lambda: "main-stream")
    return monkeypatch


def test_inline_path_keeps_wait_stream(barrier_env):
    barrier_env.delenv("VLLM_ASCEND_GRAPH_UPDATE_ASYNC", raising=False)
    stream = _FakeStream()

    assert ug.begin_update_barrier(stream) is None
    assert stream.waited == ["main-stream"]


def test_async_path_records_event_instead_of_touching_stream(barrier_env):
    barrier_env.setenv("VLLM_ASCEND_GRAPH_UPDATE_ASYNC", "1")
    stream = _FakeStream()
    events = []

    def _make_event():
        event = _FakeEvent()
        events.append(event)
        return event

    barrier_env.setattr(ug.torch.npu, "Event", _make_event)

    result = ug.begin_update_barrier(stream)

    assert stream.waited == [], "the model-runner thread must not enqueue to the update stream"
    assert events[0].recorded == ["main-stream"]
    assert result is events[0]


def test_update_dispatches_inline_by_default(barrier_env):
    barrier_env.delenv("VLLM_ASCEND_GRAPH_UPDATE_ASYNC", raising=False)
    executor = _FakeExecutor()
    barrier_env.setattr(ug, "GRAPH_UPDATE_EXECUTOR", executor)
    graph = _FakeGraph()

    ug.UpdatableGraph.update(graph, "update-stream", ("task",), None)

    assert executor.submitted == []
    assert graph.calls == [("update-stream", ("task",), None)]


def test_update_submits_to_worker_when_enabled(barrier_env):
    barrier_env.setenv("VLLM_ASCEND_GRAPH_UPDATE_ASYNC", "1")
    executor = _FakeExecutor()
    barrier_env.setattr(ug, "GRAPH_UPDATE_EXECUTOR", executor)
    graph = _FakeGraph()

    ug.UpdatableGraph.update(graph, "update-stream", ("task",), "barrier")

    assert executor.submitted == [(graph, "update-stream", ("task",), "barrier")]
    assert graph.calls == [], "the model-runner thread must not apply the update itself"


def test_worker_applies_jobs_in_submission_order():
    executor = ug.GraphUpdateExecutor()
    graph = _FakeGraph()

    for index in range(5):
        executor.submit(graph, f"stream-{index}", (index,), f"event-{index}")
    executor.wait_idle()

    assert [call[1] for call in graph.calls] == [(0,), (1,), (2,), (3,), (4,)]
    assert [call[2] for call in graph.calls] == [f"event-{i}" for i in range(5)]


def test_worker_failure_is_raised_on_the_runner_thread():
    executor = ug.GraphUpdateExecutor()

    executor.submit(_FakeGraph(fail=True), "stream", ("task",), "event")
    with pytest.raises(RuntimeError, match="worker thread failed"):
        executor.wait_idle()
    with pytest.raises(RuntimeError, match="worker thread failed"):
        executor.submit(_FakeGraph(), "stream", ("task",), "event")
