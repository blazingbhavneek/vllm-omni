# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Irodori per-stage precision policy."""

import pytest
import torch

from vllm_omni.diffusion.models.irodori_tts.model import (
    JointAttention,
    TextToLatentRFDiT,
)
from vllm_omni.diffusion.models.irodori_tts.precision import (
    IEEE,
    TF32,
    PRECISION_PROFILES,
    REFERENCE_POLICY,
    TRAINED_POLICY,
    supported_attention_dtype,
    matmul_precision,
    resolve_precision_policy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_reference_profile_is_strict_ieee():
    assert REFERENCE_POLICY.dit_matmul == IEEE
    assert REFERENCE_POLICY.codec_matmul == IEEE
    assert REFERENCE_POLICY.condition_matmul == IEEE
    assert REFERENCE_POLICY.attention_dtype is None
    assert not REFERENCE_POLICY.uses_tf32


def test_trained_profile_keeps_condition_path_exact():
    # The duration predictor rounds to an integer frame count, so the
    # condition path must stay bit-stable even in the fast profile.
    assert TRAINED_POLICY.condition_matmul == IEEE
    assert TRAINED_POLICY.dit_matmul == TF32
    assert TRAINED_POLICY.codec_matmul == TF32
    assert TRAINED_POLICY.attention_dtype is torch.bfloat16


@pytest.mark.parametrize("name", sorted(PRECISION_PROFILES))
def test_resolve_known_profiles(name):
    assert resolve_precision_policy(name) is PRECISION_PROFILES[name]


@pytest.mark.parametrize("bad", ["fastest", "", None, 3, "TRAINED"])
def test_resolve_rejects_unknown_profiles(bad):
    with pytest.raises(ValueError, match="irodori_precision_profile"):
        resolve_precision_policy(bad)


def test_resolve_passes_through_policy_objects():
    assert resolve_precision_policy(TRAINED_POLICY) is TRAINED_POLICY


def test_matmul_precision_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported matmul precision"):
        with matmul_precision("bf16x3"):
            pass


def test_matmul_precision_restores_previous_state():
    if not torch.cuda.is_available():
        pytest.skip("precision flags are only mutated on CUDA")
    read = (
        (lambda: torch.backends.cuda.matmul.fp32_precision)
        if hasattr(torch.backends.cuda.matmul, "fp32_precision")
        else (lambda: torch.backends.cuda.matmul.allow_tf32)
    )
    before = read()
    with matmul_precision(TF32):
        assert read() != before or before in (TF32, True)
    assert read() == before


def test_matmul_precision_restores_on_exception():
    if not torch.cuda.is_available():
        pytest.skip("precision flags are only mutated on CUDA")
    read = (
        (lambda: torch.backends.cuda.matmul.fp32_precision)
        if hasattr(torch.backends.cuda.matmul, "fp32_precision")
        else (lambda: torch.backends.cuda.matmul.allow_tf32)
    )
    before = read()
    with pytest.raises(RuntimeError):
        with matmul_precision(TF32):
            raise RuntimeError("boom")
    assert read() == before


def test_supported_attention_dtype_none_without_policy():
    assert supported_attention_dtype(None) is None
    assert supported_attention_dtype(REFERENCE_POLICY) is None


def test_supported_attention_dtype_requires_cuda():
    expected = torch.bfloat16 if (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ) else None
    assert supported_attention_dtype(TRAINED_POLICY) is expected


def test_dit_declares_repeated_blocks_for_regional_compile():
    assert TextToLatentRFDiT._repeated_blocks == ["DiffusionBlock"]


def test_set_precision_policy_reaches_every_joint_attention(tiny_dit):
    attentions = [m for m in tiny_dit.modules() if isinstance(m, JointAttention)]
    assert attentions, "fixture should build at least one DiT block"
    assert all(m.attention_dtype is None for m in attentions)

    tiny_dit.set_precision_policy(TRAINED_POLICY)
    expected = supported_attention_dtype(TRAINED_POLICY)
    assert all(m.attention_dtype is expected for m in attentions)

    tiny_dit.set_precision_policy(None)
    assert all(m.attention_dtype is None for m in attentions)


def test_stage_matmul_mode_defaults_to_ieee(tiny_dit):
    # An unconfigured model must behave exactly like upstream.
    assert tiny_dit._stage_matmul_mode("dit") == IEEE
    assert tiny_dit._stage_matmul_mode("condition") == IEEE
    tiny_dit.set_precision_policy(TRAINED_POLICY)
    assert tiny_dit._stage_matmul_mode("dit") == TF32
    assert tiny_dit._stage_matmul_mode("condition") == IEEE
