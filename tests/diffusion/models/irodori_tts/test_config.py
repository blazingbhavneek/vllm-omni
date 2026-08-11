# SPDX-License-Identifier: Apache-2.0
"""Metadata-only Irodori checkpoint loading tests."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.irodori_tts.config import (
    ModelConfig,
    read_irodori_checkpoint_config,
    resolve_irodori_checkpoint,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _write_checkpoint(path, config, text_config=None):
    metadata = {"config_json": json.dumps(config)}
    if text_config is not None:
        metadata["text_encoder_config_json"] = json.dumps(text_config)
    save_file({"weight": torch.zeros(1)}, str(path), metadata=metadata)


def _tiny_config():
    config = ModelConfig(
        latent_dim=4,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        text_dim=8,
        text_layers=1,
        text_heads=2,
        speaker_dim=8,
        speaker_layers=1,
        speaker_heads=2,
        duration_hidden_dim=8,
        duration_layers=1,
        duration_attention_heads=2,
    )
    return {field: getattr(config, field) for field in config.__dataclass_fields__} | {
        "max_text_len": 16,
        "max_caption_len": 24,
        "ref_max_seconds": 12.0,
    }


def test_reads_metadata_without_loading_tensors(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_checkpoint(path, _tiny_config(), {"model_type": "modernbert", "hidden_size": 8})

    config = read_irodori_checkpoint_config(path)

    assert config.model.model_dim == 8
    assert config.max_text_len == 16
    assert config.max_caption_len == 24
    assert config.max_ref_seconds == 12.0
    assert config.text_encoder_config["model_type"] == "modernbert"


@pytest.mark.parametrize(
    ("metadata_config", "text_config", "match"),
    [
        (None, {"model_type": "modernbert"}, "config_json"),
        ("not-json", {"model_type": "modernbert"}, "Invalid JSON"),
        ([], {"model_type": "modernbert"}, "must decode to an object"),
        ({**_tiny_config(), "unknown": 1}, {"model_type": "modernbert"}, "Unknown Irodori"),
        (_tiny_config(), None, "text_encoder_config_json"),
        (_tiny_config(), [], "must decode to an object"),
    ],
)
def test_rejects_bad_metadata(tmp_path, metadata_config, text_config, match):
    path = tmp_path / "model.safetensors"
    metadata = {}
    if metadata_config is not None:
        metadata["config_json"] = metadata_config if isinstance(metadata_config, str) else json.dumps(metadata_config)
    if text_config is not None:
        metadata["text_encoder_config_json"] = json.dumps(text_config)
    save_file({"weight": torch.zeros(1)}, str(path), metadata=metadata)

    with pytest.raises(ValueError, match=match):
        read_irodori_checkpoint_config(path)


def test_local_checkpoint_resolution_requires_canonical_filename(tmp_path):
    wrong = tmp_path / "checkpoint.safetensors"
    wrong.touch()
    with pytest.raises(ValueError, match="model.safetensors"):
        resolve_irodori_checkpoint(str(wrong))


def test_v4_small_pinned_values_when_local_checkpoint_is_available():
    path = "../Aratako/Irodori-TTS-v4-Small/model.safetensors"
    config = read_irodori_checkpoint_config(path)
    assert (config.model.latent_dim, config.model.model_dim, config.model.num_layers) == (32, 1280, 12)
    assert (config.model.num_heads, config.model.speaker_dim, config.model.speaker_patch_size) == (20, 768, 4)
    assert (config.max_text_len, config.max_caption_len, config.max_ref_seconds) == (256, 512, 120.0)
