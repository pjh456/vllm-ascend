# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Hashable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.utils import weak_ref_tensors

Params = dict[str, Any]


class GraphUpdateStats:
    """Host-side cost of UpdatableGraph parameter updates, per step.

    The V2 model runner re-issues every captured attention op through
    ``graph_task_update_*`` on each replay. This collects the wall time of that
    loop so the cost can be attributed to a specific phase instead of guessed
    at. Disabled unless ``VLLM_ASCEND_GRAPH_UPDATE_PROFILE=1``.
    """

    LOG_EVERY_STEPS = 100

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = envs.VLLM_ASCEND_GRAPH_UPDATE_PROFILE if enabled is None else enabled
        self._clear()

    def _clear(self) -> None:
        self.steps = 0
        self.tasks = 0
        self.resolve_s = 0.0
        self.update_s = 0.0
        self.begin_s = 0.0
        self.op_s = 0.0
        self.end_s = 0.0
        self.record_s = 0.0

    def add_step(self, resolve_s: float, update_s: float) -> None:
        self.steps += 1
        self.resolve_s += resolve_s
        self.update_s += update_s
        if self.enabled and self.steps >= self.LOG_EVERY_STEPS:
            self.report()

    def report(self) -> None:
        n = self.steps
        if n == 0:
            return
        logger.info(
            "[graph-update] steps=%d tasks/step=%.1f | resolve %.3f ms/step | "
            "update %.3f ms/step (begin %.3f, op %.3f, end %.3f, record %.3f)",
            n,
            self.tasks / n,
            1e3 * self.resolve_s / n,
            1e3 * self.update_s / n,
            1e3 * self.begin_s / n,
            1e3 * self.op_s / n,
            1e3 * self.end_s / n,
            1e3 * self.record_s / n,
        )
        self._clear()


GRAPH_UPDATE_STATS = GraphUpdateStats()


class ParamProvider(Protocol):
    def resolve(self, context) -> Params: ...


class ParamSource(Protocol):
    def get(
        self,
        provider: ParamProvider,
    ) -> Sequence[Params]: ...


@dataclass(frozen=True, slots=True)
class ContextSource:
    context: Any

    def get(
        self,
        provider: ParamProvider,
    ) -> Sequence[Params]:
        return (provider.resolve(self.context),)


@dataclass(frozen=True, slots=True)
class SharedSource:
    params: Sequence[Params]

    def get(
        self,
        _provider: ParamProvider,
    ) -> Sequence[Params]:
        layer_name = getattr(_provider, "layer_name", None)
        return [
            {k: v for k, v in param.items() if k != "layer_name"}
            for param in self.params
            if layer_name == param.get("layer_name")
        ]


_ACTIVE_GRAPH: ContextVar["UpdatableGraph | None"] = ContextVar("capturing_updatable_graph", default=None)


@dataclass(slots=True)
class GraphUpdateTask:
    operation: Callable[..., Any]
    kwargs: dict[str, Any]
    provider: ParamProvider
    provider_index: int
    handle: Any
    event: Any

    def bind(self, params: Params) -> "GraphUpdateTask":
        runtime_kwargs = {**self.kwargs, **params}
        return replace(self, kwargs=runtime_kwargs)

    def apply(self, update_stream) -> None:
        stats = GRAPH_UPDATE_STATS
        if stats.enabled:
            t0 = time.perf_counter()
            torch.npu.graph_task_update_begin(update_stream, self.handle)
            t1 = time.perf_counter()
            self.operation(**self.kwargs)
            t2 = time.perf_counter()
            torch.npu.graph_task_update_end(update_stream)
            t3 = time.perf_counter()
            self.event.record(update_stream)
            t4 = time.perf_counter()
            stats.begin_s += t1 - t0
            stats.op_s += t2 - t1
            stats.end_s += t3 - t2
            stats.record_s += t4 - t3
            stats.tasks += 1
            return
        torch.npu.graph_task_update_begin(update_stream, self.handle)
        self.operation(**self.kwargs)
        torch.npu.graph_task_update_end(update_stream)
        self.event.record(update_stream)


class UpdatableGraph(torch.npu.NPUGraph):
    def __init__(self) -> None:
        super().__init__()
        self.tasks: list[GraphUpdateTask] = []
        self.provider_sizes: dict[ParamProvider, int] = {}
        self.capture_resources: dict[Hashable, Any] = {}
        self.capture_token: Token[UpdatableGraph | None] | None = None

    def capture_begin(self, pool=None, capture_error_mode: str = "global") -> None:
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)
        assert self.capture_token is None
        self.capture_token = _ACTIVE_GRAPH.set(self)

    def capture_end(self) -> None:
        try:
            super().capture_end()
        finally:
            assert self.capture_token is not None
            _ACTIVE_GRAPH.reset(self.capture_token)
            self.capture_token = None
            self.capture_resources = weak_ref_tensors(self.capture_resources)
            if GRAPH_UPDATE_STATS.enabled:
                logger.info(
                    "[graph-update] captured graph with %d update tasks",
                    len(self.tasks),
                )

    def get_capture_resource(
        self,
        key: Hashable,
        factory: Callable[[], Any],
        use_max_workspace: bool = False,
    ) -> Any:
        if key not in self.capture_resources:
            self.capture_resources[key] = factory()
        if use_max_workspace:
            # Some models mix attention layer shapes under the same graph size.
            # During capture, keep the largest required workspace for that size.
            candidate_workspace = factory()
            if (
                candidate_workspace.numel() * candidate_workspace.element_size()
                > self.capture_resources[key].numel() * self.capture_resources[key].element_size()
            ):
                self.capture_resources[key] = candidate_workspace
        return self.capture_resources[key]

    def register_task(
        self,
        operation: Callable[..., Any],
        kwargs: dict[str, Any],
        provider: ParamProvider,
    ) -> None:
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        operation(**kwargs)
        handle = torch.npu.graph_task_group_end(stream)
        weak_kwargs = weak_ref_tensors(kwargs)
        provider_index = self.provider_sizes.get(provider, 0)
        self.provider_sizes[provider] = provider_index + 1
        self.tasks.append(
            GraphUpdateTask(
                operation,
                weak_kwargs,
                provider,
                provider_index,
                handle,
                event,
            )
        )

    def resolve_tasks(
        self,
        source: ParamSource,
    ) -> tuple[GraphUpdateTask, ...]:
        params_by_provider = {provider: source.get(provider) for provider in self.provider_sizes}
        for provider, size in self.provider_sizes.items():
            assert len(params_by_provider[provider]) == size
        return tuple(task.bind(params_by_provider[task.provider][task.provider_index]) for task in self.tasks)

    def update(
        self,
        update_stream,
        resolved_tasks: tuple[GraphUpdateTask, ...],
    ) -> None:
        logger.debug_once("Updating host-side attention metadata with UpdatableGraph.")
        with torch.npu.stream(update_stream):
            # This is specially designed for PA.
            ws_buffer: dict[Hashable, Any] = {}
            for task in resolved_tasks:
                if task.operation == torch_npu._npu_paged_attention:
                    ws_key = _get_ws_key(task.kwargs)
                    if ws_buffer.get(ws_key) is None:
                        ws_kwargs = task.kwargs.copy()
                        ws_kwargs.pop("workspace")
                        workspace = torch_npu._npu_paged_attention_get_workspace(**ws_kwargs)
                        ws_buffer[ws_key] = workspace
                    task.kwargs["workspace"] = ws_buffer[ws_key]
                task.apply(update_stream)


def register_task(
    operation: Callable[..., Any],
    kwargs: dict[str, Any],
    provider: ParamProvider,
) -> None:
    graph = _ACTIVE_GRAPH.get()
    if graph is None:
        operation(**kwargs)
    else:
        graph.register_task(operation, kwargs, provider)


def get_capture_resource(
    key: Hashable,
    factory: Callable[[], Any],
    use_max_workspace: bool = False,
) -> Any:
    graph = _ACTIVE_GRAPH.get()
    if graph is None:
        return factory()
    return graph.get_capture_resource(key, factory, use_max_workspace)


def _get_ws_key(kwargs):
    def _sig(t: Any) -> tuple | None:
        if t is None:
            return (None, None)
        return (t.shape, t.dtype)

    return (
        kwargs["context_lens"].data_ptr(),
        tuple(kwargs["context_lens"].shape),
        _sig(kwargs["query"]),
        _sig(kwargs["key_cache"]),
        _sig(kwargs["value_cache"]),
        _sig(kwargs["block_table"]),
        _sig(kwargs["out"]),
        kwargs["num_kv_heads"],
        kwargs["num_heads"],
        kwargs["scale_value"],
    )
