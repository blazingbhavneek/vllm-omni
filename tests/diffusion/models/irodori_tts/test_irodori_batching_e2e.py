# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end validation that Irodori-TTS really batches multiple requests.

``test_irodori_packed_batching.py`` proves the packed batch is row-independent
against a stub DiT. These tests drive the real model through the offline Python
entrypoint and check the properties only the full stack can show.

What is exact, and asserted as such:

* a one-request wave is **bit-identical** to a serial render — the batching code
  path itself introduces nothing;
* identical requests inside one wave are **bit-identical to each other** — no
  cross-request contamination through shared padding or shared context K/V;
* every render has the exact requested sample count;
* each request in a wave still resolves to *its own* audio, not a neighbour's.

What is *not* exact, and is deliberately not asserted tightly: a wave of two or
more requests changes the numerics relative to a serial render. Packed
attention runs in bfloat16 at every parameter dtype, so a different wave size
retiles the fused GEMM/attention and shifts results at the last bits; the
rectified-flow ODE then amplifies that over the denoise steps. Measured on an
RTX 5060 Ti for a 4 s render, relative RMSE against the serial render is
0.03-0.55 depending on the text, and it does not shrink at 40 steps or with
float32 parameters. The audio is a different valid sample, not a corrupted one.
``_report_similarity`` prints the numbers so a reviewer can see the drift; only
a loose ceiling is enforced, to catch outright garbage.

``diffusion_batch_size`` is what admits more than one request into a denoise
step (it becomes ``OmniDiffusionConfig.max_num_seqs``, which the step scheduler
reads as ``max_num_running_reqs``). It is a Python-only knob today: the
``vllm serve`` path pins it to 1, so these tests use ``AsyncOmni`` rather than
the HTTP server.

Requires a GPU and the ``irodori-tts`` extra. Point at an already-downloaded
checkpoint directory to avoid a Hub round-trip::

    VLLM_OMNI_IRODORI_TEST_MODEL=/path/to/Irodori-TTS-v4-Small \
        pytest -q tests/diffusion/models/irodori_tts/test_irodori_batching_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, pytest.mark.tts]

MODEL = os.environ.get("VLLM_OMNI_IRODORI_TEST_MODEL", "Aratako/Irodori-TTS-v4-Small")
SAMPLE_RATE = 48_000
# Kept small: these tests check batching semantics, not audio quality.
NUM_STEPS = 8
CAPTION = "落ち着いた女性の声"

# Distinct text *and* distinct seed per request, so every request has a unique
# waveform and a swap between requests is detectable.
TEXTS = (
    "こんにちは。これは音声合成のテストです。",
    "今日はとても良い天気ですね。",
    "バッチ処理の検証を行っています。",
    "音声の品質を確認してください。",
)
SEEDS = (1729, 4104, 13832, 20683)

# Loose ceiling: a wave render must still be recognisably the same request, but
# see the module docstring for why this cannot be tight.
MAX_WAVE_RELATIVE_RMSE = 1.0
MIN_WAVE_CORRELATION = 0.5


def _sampling(seed: int, seconds: float) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        seed=seed,
        num_inference_steps=NUM_STEPS,
        extra_args={"seconds": seconds},
    )


def _extract_audio(output) -> np.ndarray:
    payload = getattr(output, "multimodal_output", None)
    if payload is None:
        request_output = getattr(output, "request_output", None)
        payload = getattr(request_output, "multimodal_output", None)
    if not isinstance(payload, dict) or "audio" not in payload:
        raise AssertionError("Irodori produced no audio output.")
    audio = np.asarray(payload["audio"], dtype=np.float32).squeeze()
    sample_rate = int(payload.get("audio_sample_rate", payload.get("sr", 0)))
    assert sample_rate == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sample_rate}"
    assert audio.ndim == 1 and audio.size, f"unexpected audio shape {audio.shape}"
    assert np.isfinite(audio).all(), "audio contains non-finite samples"
    return audio


def _similarity(generated: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    assert generated.shape == reference.shape, f"shape drift {generated.shape} vs {reference.shape}"
    relative_rmse = float(np.sqrt(np.mean((generated - reference) ** 2)) / max(np.sqrt(np.mean(reference**2)), 1e-8))
    generated_centered = generated - generated.mean()
    reference_centered = reference - reference.mean()
    correlation = float(
        np.dot(generated_centered, reference_centered)
        / max(np.linalg.norm(generated_centered) * np.linalg.norm(reference_centered), 1e-8)
    )
    return relative_rmse, correlation


def _assert_sample_counts(audios, plan) -> None:
    for audio, (index, seconds) in zip(audios, plan, strict=True):
        expected = int(round(seconds * SAMPLE_RATE))
        assert audio.shape[0] == expected, f"request {index}: expected {expected} samples, got {audio.shape[0]}"


def _report_similarity(label: str, wave, serial, plan) -> None:
    """Print wave-vs-serial drift and enforce only a loose ceiling."""
    for position, (index, seconds) in enumerate(plan):
        relative_rmse, correlation = _similarity(wave[position], serial[position])
        print(f"   [{label}] request {index} ({seconds}s): rel_rmse={relative_rmse:.6f} corr={correlation:.8f}")
        assert relative_rmse <= MAX_WAVE_RELATIVE_RMSE and correlation >= MIN_WAVE_CORRELATION, (
            f"request {index} diverged beyond the wave-numerics ceiling: "
            f"rel_rmse={relative_rmse:.6f} corr={correlation:.8f}"
        )


def _assert_identity_preserved(wave, serial, plan) -> None:
    """Each wave render must resemble its own serial render more than any other."""
    for position, (index, _) in enumerate(plan):
        scores = {}
        for other, (other_index, _) in enumerate(plan):
            if serial[other].shape != wave[position].shape:
                continue
            scores[other_index] = _similarity(wave[position], serial[other])[1]
        best = max(scores, key=scores.get)
        assert best == index, (
            f"wave request {index} best-matches serial request {best} "
            f"(corr {scores[best]:.6f} vs own {scores[index]:.6f}); requests are being crossed"
        )


async def _render(omni: AsyncOmni, index: int, seconds: float, tag: str) -> np.ndarray:
    last = None
    async for output in omni.generate(
        prompt={"input": TEXTS[index], "caption": CAPTION},
        request_id=f"irodori-{tag}-{index}-{time.monotonic_ns()}",
        sampling_params_list=[_sampling(SEEDS[index], seconds)],
    ):
        last = output
    assert last is not None, f"no output for request {index}"
    return _extract_audio(last)


async def _render_serial(omni: AsyncOmni, plan) -> tuple[list[np.ndarray], float]:
    """One request in flight at a time — the batch-size-1 reference."""
    started = time.perf_counter()
    audios = [await _render(omni, index, seconds, "serial") for index, seconds in plan]
    return audios, time.perf_counter() - started


async def _render_wave(omni: AsyncOmni, plan) -> tuple[list[np.ndarray], float]:
    """All requests in flight together — the step scheduler co-schedules them."""
    started = time.perf_counter()
    audios = await asyncio.gather(*(_render(omni, index, seconds, "wave") for index, seconds in plan))
    return list(audios), time.perf_counter() - started


def _engine(batch_size: int) -> AsyncOmni:
    return AsyncOmni(
        model=MODEL,
        model_class_name="IrodoriTTSPipeline",
        max_num_seqs=batch_size,
        diffusion_batch_size=batch_size,
        step_execution=True,
    )


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_single_request_wave_is_bit_identical_to_serial():
    """With one request in flight the batching path must change nothing."""

    async def _inner():
        omni = _engine(4)
        try:
            serial = await _render(omni, 0, 4.0, "serial")
            wave, _ = await _render_wave(omni, [(0, 4.0)])
        finally:
            omni.shutdown()
        np.testing.assert_array_equal(wave[0], serial)

    asyncio.run(_inner())


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_identical_requests_in_one_wave_are_bit_identical():
    """No cross-request contamination: same input in a wave gives same output."""
    plan = [(0, 4.0)] * 4

    async def _inner():
        omni = _engine(len(plan))
        try:
            wave, _ = await _render_wave(omni, plan)
        finally:
            omni.shutdown()
        _assert_sample_counts(wave, plan)
        for position in range(1, len(wave)):
            np.testing.assert_array_equal(wave[position], wave[0])

    asyncio.run(_inner())


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_homogeneous_wave_preserves_request_identity():
    """Four same-length, different-content requests must not be crossed."""
    plan = [(index, 4.0) for index in range(len(TEXTS))]

    async def _inner():
        omni = _engine(len(plan))
        try:
            serial, serial_s = await _render_serial(omni, plan)
            wave, wave_s = await _render_wave(omni, plan)
        finally:
            omni.shutdown()
        print(f"\n[homogeneous] serial={serial_s:.2f}s wave={wave_s:.2f}s (unwarmed; see the throughput test)")
        _assert_sample_counts(wave, plan)
        _assert_identity_preserved(wave, serial, plan)
        _report_similarity("homogeneous", wave, serial, plan)

    asyncio.run(_inner())


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_mixed_length_wave_preserves_request_identity():
    """Mixed durations exercise ragged/padded grouping rather than one bucket."""
    plan = [(0, 4.0), (1, 10.0), (2, 4.0), (3, 10.0)]

    async def _inner():
        omni = _engine(len(plan))
        try:
            serial, serial_s = await _render_serial(omni, plan)
            wave, wave_s = await _render_wave(omni, plan)
        finally:
            omni.shutdown()
        print(f"\n[mixed-length] serial={serial_s:.2f}s wave={wave_s:.2f}s (unwarmed; see the throughput test)")
        _assert_sample_counts(wave, plan)
        _assert_identity_preserved(wave, serial, plan)
        _report_similarity("mixed-length", wave, serial, plan)

    asyncio.run(_inner())


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_batching_beats_serial_execution():
    """The wave must be faster than the same requests run one at a time.

    This is the regression guard for the failure mode where every layer of the
    batching stack is present and correct but the scheduler still admits one
    request per denoise step, making concurrency a no-op.

    Both paths must be warmed before either is timed. Regional ``torch.compile``
    compiles the wave's batch shapes on their first use, and charging that to
    the wave inverts the result — an unwarmed wave measured 0.40x where a warmed
    one measures 1.7x.
    """
    plan = [(index, 4.0) for index in range(len(TEXTS))]

    async def _inner():
        omni = _engine(len(plan))
        try:
            await _render_serial(omni, plan)
            await _render_wave(omni, plan)
            await _render_wave(omni, plan)
            serial, serial_s = await _render_serial(omni, plan)
            wave, wave_s = await _render_wave(omni, plan)
        finally:
            omni.shutdown()
        speedup = serial_s / wave_s if wave_s > 0 else float("inf")
        print(f"\n[throughput] serial={serial_s:.2f}s wave={wave_s:.2f}s speedup={speedup:.2f}x")
        assert len(wave) == len(serial) == len(plan)
        assert wave_s < serial_s, (
            f"batched wave ({wave_s:.2f}s) was not faster than serial ({serial_s:.2f}s) "
            f"for {len(plan)} requests; requests are most likely executing one per denoise step"
        )

    asyncio.run(_inner())
