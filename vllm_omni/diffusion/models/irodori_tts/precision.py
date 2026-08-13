# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-stage matmul precision policy for Irodori-TTS.

The released ``Aratako/Irodori-TTS-v4-Small`` checkpoint stores FP32 master
weights, but it was *trained* under BF16 autocast with TF32 matmuls enabled
(``configs/train_v4_small.yaml``: ``precision: bf16``, ``allow_tf32: true``;
``train.py`` wraps the whole forward in ``torch.autocast(cuda, bfloat16)`` and
prints "Compute precision=bf16 (weights/optimizer states kept in fp32)").
Running inference under strict IEEE FP32 therefore evaluates the model in a
*stricter* numerical regime than it ever saw during training, at roughly half
the achievable throughput.

This module expresses that as an explicit per-stage policy instead of one
global dtype, so the stages that must stay bit-stable can, while the heavy
stages use the datapaths the weights were fitted under.

``reference``
    Everything IEEE FP32.  The frozen parity baseline; keep this runnable.

``trained`` (default)
    TF32 matmuls in the DiT denoise loop and the DACVAE codec, BF16 joint
    attention inside the DiT, and IEEE FP32 for the condition encoders and the
    duration predictor.

The condition/duration carve-out is not cosmetic.  ``_duration_lengths()``
does ``round(expm1(prediction).mean() * duration_scale)``; a rounding flip
changes the latent shape, hence the noise draw, hence the entire render.  The
measured TF32 perturbation there is ~0.02 codec frames, which would flip a few
percent of requests.  Duration prediction costs ~1 ms out of a ~5 s request,
so pinning it to IEEE FP32 removes that discrete risk for free.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch

# Matmul modes accepted by ``matmul_precision``.
IEEE = "ieee"
TF32 = "tf32"


@dataclass(frozen=True)
class IrodoriPrecisionPolicy:
    """Which datapath each Irodori stage may use."""

    name: str
    dit_matmul: str
    codec_matmul: str
    condition_matmul: str
    # ``None`` keeps joint attention in the activation dtype.
    attention_dtype: torch.dtype | None

    @property
    def uses_tf32(self) -> bool:
        return TF32 in (self.dit_matmul, self.codec_matmul, self.condition_matmul)


REFERENCE_POLICY = IrodoriPrecisionPolicy(
    name="reference",
    dit_matmul=IEEE,
    codec_matmul=IEEE,
    condition_matmul=IEEE,
    attention_dtype=None,
)

TRAINED_POLICY = IrodoriPrecisionPolicy(
    name="trained",
    dit_matmul=TF32,
    codec_matmul=TF32,
    # Condition encoders and the duration predictor feed an integer rounding
    # decision, so they stay bit-stable.
    condition_matmul=IEEE,
    attention_dtype=torch.bfloat16,
)

PRECISION_PROFILES: dict[str, IrodoriPrecisionPolicy] = {
    REFERENCE_POLICY.name: REFERENCE_POLICY,
    TRAINED_POLICY.name: TRAINED_POLICY,
}


def resolve_precision_policy(profile: object) -> IrodoriPrecisionPolicy:
    if isinstance(profile, IrodoriPrecisionPolicy):
        return profile
    if not isinstance(profile, str) or profile not in PRECISION_PROFILES:
        raise ValueError(
            "irodori_precision_profile must be one of "
            f"{sorted(PRECISION_PROFILES)}, got {profile!r}."
        )
    return PRECISION_PROFILES[profile]


def _read_matmul_state() -> tuple[object, ...]:
    """Snapshot whichever precision knobs this torch build exposes."""
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        return (
            torch.backends.cuda.matmul.fp32_precision,
            torch.backends.cudnn.conv.fp32_precision,
        )
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )


def _write_matmul_state(state: tuple[object, ...]) -> None:
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = state[0]
        torch.backends.cudnn.conv.fp32_precision = state[1]
        return
    torch.backends.cuda.matmul.allow_tf32 = state[0]
    torch.backends.cudnn.allow_tf32 = state[1]
    torch.set_float32_matmul_precision(state[2])


def _apply_mode(mode: str) -> None:
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.conv.fp32_precision = mode
        return
    enabled = mode == TF32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


@contextlib.contextmanager
def matmul_precision(mode: str) -> Iterator[None]:
    """Scope the process-wide FP32 matmul/conv precision to ``mode``.

    These backend flags are global, so the previous values are restored on
    exit.  Each diffusion worker runs its model on one thread, so scoping them
    per stage is safe there; this is not a general-purpose reentrant guard.
    """
    if mode not in (IEEE, TF32):
        raise ValueError(f"Unsupported matmul precision {mode!r}.")
    if not torch.cuda.is_available():
        yield
        return
    previous = _read_matmul_state()
    _apply_mode(mode)
    try:
        yield
    finally:
        _write_matmul_state(previous)


def supported_attention_dtype(
    policy: IrodoriPrecisionPolicy | None,
) -> torch.dtype | None:
    """Resolve the policy's attention dtype against this device, once.

    FP32 SDPA has no FlashAttention kernel and ignores TF32 entirely -- it
    falls back to a SIMT CUTLASS mem-efficient kernel -- so downcasting the
    SDPA call is the only lever that speeds joint attention up.

    This is deliberately resolved at policy-install time rather than per
    forward: the hardware queries below are not traceable, and
    ``JointAttention.forward`` runs inside the regionally compiled region.
    """
    if policy is None or policy.attention_dtype is None:
        return None
    if not torch.cuda.is_available():
        return None
    if policy.attention_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return None
    return policy.attention_dtype
