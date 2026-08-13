# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared tiny-model fixtures for the Irodori-TTS unit tests."""

import pytest

from vllm_omni.diffusion.models.irodori_tts.config import ModelConfig
from vllm_omni.diffusion.models.irodori_tts.model import TextToLatentRFDiT


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    """A minimal v4-shaped config: caption + speaker on, scratch text encoder."""
    return ModelConfig(
        latent_dim=8,
        latent_patch_size=1,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        text_vocab_size=64,
        text_dim=16,
        text_layers=1,
        text_heads=2,
        use_caption_condition=True,
        caption_vocab_size=64,
        caption_dim=16,
        caption_layers=1,
        caption_heads=2,
        use_speaker_condition=True,
        speaker_dim=16,
        speaker_layers=1,
        speaker_heads=2,
        speaker_patch_size=2,
        timestep_embed_dim=16,
        adaln_rank=8,
    )


@pytest.fixture
def tiny_dit(tiny_model_config: ModelConfig) -> TextToLatentRFDiT:
    return TextToLatentRFDiT(tiny_model_config)
