# SPDX-License-Identifier: Apache-2.0
"""Context bucket selection and prefix-length precomputation tests."""

import types

import pytest
import torch

from vllm_omni.diffusion.models.irodori_tts.batching import (
    IrodoriContextBucketPolicy,
    IrodoriDenoiseBatch,
    _source_bucket,
)
from vllm_omni.diffusion.models.irodori_tts.pipeline_irodori_tts import IrodoriTTSPipeline
from vllm_omni.diffusion.models.irodori_tts.sampler import (
    _bundle_prefix_lengths,
    _conditional_rows,
    run_packed_euler_rf_cfg_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

POLICY = IrodoriContextBucketPolicy((8, 16, 32, 64))


def _mask(valid: int, total: int, rows: int = 1) -> torch.Tensor:
    mask = torch.zeros((rows, total), dtype=torch.bool)
    mask[:, :valid] = True
    return mask


def _bundle(text=(5, 16), speaker=None, caption=None):
    def source(spec, dim):
        if spec is None:
            return None, None
        valid, total = spec
        return torch.zeros((1, total, dim)), _mask(valid, total)

    text_state, text_mask = source(text, 4)
    speaker_state, speaker_mask = source(speaker, 4)
    caption_state, caption_mask = source(caption, 4)
    return (text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask)


def test_prefix_lengths_read_the_last_valid_position():
    assert _bundle_prefix_lengths(_bundle(text=(5, 16))) == (5, None, None)
    assert _bundle_prefix_lengths(_bundle(text=(5, 16), speaker=(30, 64))) == (5, 30, None)
    assert _bundle_prefix_lengths(_bundle(text=(1, 16), caption=(16, 16))) == (1, None, 16)


def test_prefix_length_of_a_fully_masked_source_is_one():
    """CFG uncond branches zero the mask; the bucket must stay positive."""
    bundle = _bundle(text=(0, 16))
    assert _bundle_prefix_lengths(bundle) == (1, None, None)


def test_prefix_length_falls_back_to_the_padded_width_without_a_mask():
    state = torch.zeros((1, 12, 4))
    assert _bundle_prefix_lengths((state, None, None, None, None, None)) == (12, None, None)


def test_prefix_lengths_union_across_cfg_rows():
    """An independent bundle stacks cond and uncond rows; the union wins."""
    total = 16
    mask = torch.zeros((3, total), dtype=torch.bool)
    mask[0, :7] = True  # cond row
    # rows 1 and 2 are uncond and fully masked out
    bundle = (torch.zeros((3, total, 4)), mask, None, None, None, None)
    assert _bundle_prefix_lengths(bundle) == (7, None, None)


def test_source_bucket_takes_the_largest_request_in_the_batch():
    assert _source_bucket([5, 9, 3], policy=POLICY) == (16, False)
    assert _source_bucket([5, None, 33], policy=POLICY) == (64, False)


def test_source_bucket_ignores_absent_sources():
    assert _source_bucket([None, None], policy=POLICY) == (1, False)


def test_source_bucket_marks_overflow_past_the_last_bucket():
    bucket, dynamic = _source_bucket([100], policy=POLICY)
    assert dynamic is True
    assert bucket >= 100 and bucket % 64 == 0


# --------------------------------------------------------------------------
# CFG branch reuse
# --------------------------------------------------------------------------

REQUESTS = 2
CFG_ROWS = 3  # cond + two unconditional branches
LATENT_LEN = 4
LATENT_DIM = 2


class _StubDiT:
    """Deterministic stand-in whose output depends on the text context."""

    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.row_counts: list[int] = []

    def forward_with_encoded_conditions(self, *, x_t, t, text_state, **_):
        self.row_counts.append(int(x_t.shape[0]))
        marker = text_state[:, :1, :1].reshape(-1, 1, 1)
        return x_t * 2.0 + marker + t.reshape(-1, 1, 1)


def _packed_batch(*, cfg_refresh: bool, correction: torch.Tensor | None = None):
    rows = REQUESTS * CFG_ROWS
    # Row i*CFG_ROWS is request i's conditional branch; give every row a
    # distinct marker so a wrong row selection cannot go unnoticed.
    text_state = torch.arange(rows, dtype=torch.float32).reshape(rows, 1, 1).expand(rows, 2, 2)
    bundle = (text_state.contiguous(), torch.ones((rows, 2), dtype=torch.bool), None, None, None, None)
    latents = torch.arange(
        REQUESTS * LATENT_LEN * LATENT_DIM, dtype=torch.float32
    ).reshape(REQUESTS, LATENT_LEN, LATENT_DIM)
    return IrodoriDenoiseBatch(
        request_ids=tuple(f"r{i}" for i in range(REQUESTS)),
        cfg_active=True,
        cfg_layout=("cond", "text", "speaker"),
        latents=latents,
        latent_mask=torch.ones((REQUESTS, LATENT_LEN), dtype=torch.bool),
        timesteps=torch.zeros(REQUESTS),
        dt=torch.full((REQUESTS,), 0.5),
        cfg_scales=torch.tensor([[2.0, 3.0], [2.0, 3.0]]),
        bundle=bundle,
        context_kv_cache=None,
        context_buckets=(2, 1, 1),
        dynamic_context_buckets=(False, False, False),
        is_dynamic_latent_bucket=False,
        cfg_refresh=cfg_refresh,
        cfg_correction=(
            torch.zeros_like(latents) if cfg_refresh else correction
        ),
    )


def test_conditional_rows_selects_each_request_s_conditional_branch():
    rows = REQUESTS * CFG_ROWS
    value = torch.arange(rows, dtype=torch.float32).reshape(rows, 1, 1)
    bundle = (value, None, None, None, None, None)
    selected = _conditional_rows(bundle, REQUESTS, CFG_ROWS)
    assert selected[0].flatten().tolist() == [0.0, 3.0]


def test_conditional_rows_rejects_a_row_count_that_does_not_match_the_layout():
    value = torch.zeros((5, 1, 1))
    with pytest.raises(ValueError, match="row count"):
        _conditional_rows((value, None, None, None, None, None), REQUESTS, CFG_ROWS)


def test_refresh_step_runs_every_branch_and_stores_the_correction():
    model = _StubDiT()
    batch = _packed_batch(cfg_refresh=True)
    next_latents = run_packed_euler_rf_cfg_step(model, batch)

    assert model.row_counts == [REQUESTS * CFG_ROWS]
    # marker(cond) - marker(uncond_i) is -1 and -2, scaled by 2.0 and 3.0.
    expected_correction = torch.full_like(batch.latents, 2.0 * -1.0 + 3.0 * -2.0)
    torch.testing.assert_close(batch.cfg_correction, expected_correction)

    conditional = batch.latents * 2.0 + torch.tensor([[[0.0]], [[3.0]]])
    expected = batch.latents + (conditional + expected_correction) * 0.5
    torch.testing.assert_close(next_latents, expected)


def test_reuse_step_runs_only_the_conditional_branch():
    model = _StubDiT()
    correction = torch.full((REQUESTS, LATENT_LEN, LATENT_DIM), -7.0)
    batch = _packed_batch(cfg_refresh=False, correction=correction)
    next_latents = run_packed_euler_rf_cfg_step(model, batch)

    # One row per request instead of one row per CFG branch: that is the saving.
    assert model.row_counts == [REQUESTS]

    conditional = batch.latents * 2.0 + torch.tensor([[[0.0]], [[3.0]]])
    expected = batch.latents + (conditional + correction) * 0.5
    torch.testing.assert_close(next_latents, expected)


def test_reuse_step_without_a_cached_correction_is_rejected():
    batch = _packed_batch(cfg_refresh=False, correction=None)
    with pytest.raises(ValueError, match="cached correction"):
        run_packed_euler_rf_cfg_step(_StubDiT(), batch)


def _refresh(interval: int, step_index: int, *, has_correction: bool = True, cfg_active: bool = True):
    state = types.SimpleNamespace(
        step_index=step_index,
        cfg_correction=torch.zeros(1) if has_correction else None,
    )
    return IrodoriTTSPipeline._needs_cfg_refresh(state, cfg_active, interval)


def test_interval_one_always_refreshes():
    assert all(_refresh(1, step) for step in range(6))


def test_interval_four_refreshes_every_fourth_step():
    assert [_refresh(4, step) for step in range(8)] == [
        True, False, False, False, True, False, False, False
    ]


def test_a_request_without_a_carried_correction_always_refreshes():
    assert _refresh(4, 5, has_correction=False) is True


def test_steps_with_cfg_disabled_never_reuse():
    assert _refresh(4, 5, cfg_active=False) is True


def _sampling_options(extra: dict):
    """Drive the real option parser without building a pipeline."""
    pipeline = types.SimpleNamespace(
        batching_config=types.SimpleNamespace(cfg_refresh_interval=1),
        EXTRA_BODY_PARAMS=IrodoriTTSPipeline.EXTRA_BODY_PARAMS,
        _positive_int=IrodoriTTSPipeline._positive_int,
        _finite_float=IrodoriTTSPipeline._finite_float,
    )
    sampling = types.SimpleNamespace(
        extra_args=extra, num_inference_steps=None, seed=None, generator=None, output_type="np"
    )
    return IrodoriTTSPipeline._sampling_options(pipeline, sampling)


def test_cfg_refresh_interval_survives_the_request_option_allowlist():
    """Regression: every allowlist on the request path must know this key."""
    assert "cfg_refresh_interval" in IrodoriTTSPipeline.EXTRA_BODY_PARAMS
    assert _sampling_options({"cfg_refresh_interval": 4})["cfg_refresh_interval"] == 4


def test_cfg_refresh_interval_defaults_to_the_server_setting():
    assert _sampling_options({})["cfg_refresh_interval"] == 1


def test_cfg_refresh_interval_rejects_a_non_positive_value():
    with pytest.raises(ValueError, match="cfg_refresh_interval"):
        _sampling_options({"cfg_refresh_interval": 0})


def test_adapter_forwards_cfg_refresh_interval_to_the_pipeline():
    from vllm_omni.entrypoints.openai.tts_adapters.irodori_tts import (
        _FORWARDED_EXTRAS,
        _NUMERIC_OPTIONS,
    )

    assert "cfg_refresh_interval" in _FORWARDED_EXTRAS
    assert "cfg_refresh_interval" in _NUMERIC_OPTIONS
