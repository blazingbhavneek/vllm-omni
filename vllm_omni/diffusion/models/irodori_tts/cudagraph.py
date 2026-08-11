# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded fixed-buffer CUDA graph replay for one Irodori denoise step."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .batching import IrodoriDenoiseBatch, IrodoriGraphKey
from .model import TextToLatentRFDiT

logger = init_logger(__name__)

PackedStep = Callable[[TextToLatentRFDiT, IrodoriDenoiseBatch], torch.Tensor]


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_batch: IrodoriDenoiseBatch
    static_output: torch.Tensor
    composition_key: tuple[str, ...]
    logical_bytes: int
    dynamic_shape: bool


class IrodoriCUDAGraphRunner:
    """Capture hot fixed Irodori shapes and safely fall back to eager execution."""

    def __init__(
        self,
        *,
        enabled: bool,
        batch_sizes: tuple[int, ...],
        max_entries: int,
        max_dynamic_entries: int,
        min_hits: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.batch_sizes = frozenset(int(value) for value in batch_sizes)
        self.max_entries = int(max_entries)
        self.max_dynamic_entries = int(max_dynamic_entries)
        self.min_hits = int(min_hits)
        self._cache: OrderedDict[IrodoriGraphKey, _GraphEntry] = OrderedDict()
        self._key_hits: dict[IrodoriGraphKey, int] = {}
        self._failed_keys: set[IrodoriGraphKey] = set()
        self._stats = {
            "eager": 0,
            "misses": 0,
            "hits": 0,
            "captures": 0,
            "capture_failures": 0,
            "evictions": 0,
        }
        self._memory_device: torch.device | None = None
        self._memory_baseline: tuple[int, int] | None = None
        self.last_call_info: dict[str, object] = {}

    def __call__(
        self,
        model: TextToLatentRFDiT,
        batch: IrodoriDenoiseBatch,
        eager_step: PackedStep,
    ) -> torch.Tensor:
        key = batch.graph_key
        reason = self._ineligible_reason(batch, key)
        if reason is not None:
            return self._run_eager(model, batch, eager_step, reason=reason)

        entry = self._cache.get(key)
        if entry is None:
            self._stats["misses"] += 1
            count = self._key_hits.get(key, 0) + 1
            self._key_hits[key] = count
            if count < self.min_hits:
                return self._run_eager(model, batch, eager_step, reason="warming")
            entry = self._capture(model, batch, eager_step)
            if entry is None:
                self._failed_keys.add(key)
                return self._run_eager(model, batch, eager_step, reason="capture_failed")
            self._evict_for_new_entry()
            self._cache[key] = entry
            self._stats["captures"] += 1
            mode = "capture"
        else:
            self._cache.move_to_end(key)
            mode = "hit"

        entry.static_batch.copy_dynamic_from(batch)
        if entry.composition_key != batch.composition_key:
            entry.static_batch.copy_static_from(batch)
            entry.composition_key = batch.composition_key
        entry.graph.replay()
        self._stats["hits"] += 1
        self.last_call_info = {
            "mode": "graph",
            "reason": mode,
            "key": key,
            "cache_size": len(self._cache),
        }
        # A later replay overwrites the graph-owned output buffer.
        return entry.static_output.clone()

    def _ineligible_reason(
        self,
        batch: IrodoriDenoiseBatch,
        key: IrodoriGraphKey,
    ) -> str | None:
        if not self.enabled:
            return "disabled"
        if batch.latents.device.type != "cuda":
            return "non_cuda"
        if torch.cuda.is_current_stream_capturing():
            return "nested_capture"
        if len(batch.request_ids) not in self.batch_sizes:
            return "batch_size"
        if key in self._failed_keys:
            return "failed_key"
        if batch.is_dynamic_graph_shape and key not in self._cache:
            dynamic_count = sum(entry.dynamic_shape for entry in self._cache.values())
            if dynamic_count >= self.max_dynamic_entries:
                return "dynamic_limit"
        return None

    def _run_eager(
        self,
        model: TextToLatentRFDiT,
        batch: IrodoriDenoiseBatch,
        eager_step: PackedStep,
        *,
        reason: str,
    ) -> torch.Tensor:
        self._stats["eager"] += 1
        self.last_call_info = {
            "mode": "eager",
            "reason": reason,
            "key": batch.graph_key,
            "cache_size": len(self._cache),
        }
        return eager_step(model, batch)

    def _capture(
        self,
        model: TextToLatentRFDiT,
        batch: IrodoriDenoiseBatch,
        eager_step: PackedStep,
    ) -> _GraphEntry | None:
        static_batch = batch.clone()
        try:
            if self._memory_baseline is None:
                self._memory_device = batch.latents.device
                self._memory_baseline = (
                    torch.cuda.memory_allocated(self._memory_device),
                    torch.cuda.memory_reserved(self._memory_device),
                )
            with torch.inference_mode():
                _ = eager_step(model, static_batch)
            torch.accelerator.synchronize(batch.latents.device)
            graph = torch.cuda.CUDAGraph()
            with (
                torch.inference_mode(),
                torch.cuda.graph(
                    graph,
                    pool=current_platform.get_global_graph_pool(),
                ),
            ):
                static_output = eager_step(model, static_batch)
        except Exception:
            self._stats["capture_failures"] += 1
            logger.warning(
                "Irodori CUDA graph capture failed for key=%s; this shape will remain eager.",
                batch.graph_key,
                exc_info=True,
            )
            return None

        logical_bytes = static_batch.tensor_bytes() + (static_output.numel() * static_output.element_size())
        logger.info(
            "Captured Irodori CUDA graph for key=%s (logical buffers %.2f MiB).",
            batch.graph_key,
            logical_bytes / (1024 * 1024),
        )
        return _GraphEntry(
            graph=graph,
            static_batch=static_batch,
            static_output=static_output,
            composition_key=batch.composition_key,
            logical_bytes=logical_bytes,
            dynamic_shape=batch.is_dynamic_graph_shape,
        )

    def _evict_for_new_entry(self) -> None:
        while len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1

    def clear(self) -> None:
        self._cache.clear()
        self._key_hits.clear()
        self._failed_keys.clear()
        self._memory_device = None
        self._memory_baseline = None
        self.last_call_info = {}

    def stats(self) -> dict[str, int]:
        allocated_delta = 0
        reserved_delta = 0
        if self._memory_device is not None and self._memory_baseline is not None:
            allocated_delta = (
                torch.cuda.memory_allocated(self._memory_device)
                - self._memory_baseline[0]
            )
            reserved_delta = (
                torch.cuda.memory_reserved(self._memory_device)
                - self._memory_baseline[1]
            )
        return {
            **self._stats,
            "entries": len(self._cache),
            "failed_keys": len(self._failed_keys),
            "logical_bytes": sum(entry.logical_bytes for entry in self._cache.values()),
            "cuda_allocated_delta_bytes": allocated_delta,
            "cuda_reserved_delta_bytes": reserved_delta,
        }
