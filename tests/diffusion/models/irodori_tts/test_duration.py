# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.diffusion.models.irodori_tts.duration import build_duration_features

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_duration_features_cover_all_fourteen_inputs():
    features = build_duration_features(
        ["かな漢字ABC。、「ー…!?😊"], token_counts=[10], max_text_len=20, has_speaker=[True]
    )
    assert features.shape == (1, 14)
    assert features[0, -1].item() == 1.0
    assert features[0, 3:10].gt(0).all()


def test_duration_feature_lengths_must_agree():
    with pytest.raises(ValueError, match="matching lengths"):
        build_duration_features(["text"], token_counts=[], max_text_len=4, has_speaker=[False])
