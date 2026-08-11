# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.diffusion.models.irodori_tts.tokenizer import PretrainedTextTokenizer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    eos_token = "</s>"
    pad_token_id = 0
    padding_side = "left"

    def __len__(self):
        return 10

    def encode(self, text, add_special_tokens=False):
        return [3 + idx for idx, _ in enumerate(text)]

    def __call__(self, texts, **kwargs):
        max_length = kwargs["max_length"]
        rows = [self.encode(text)[:max_length] + [0] * max(0, max_length - len(text)) for text in texts]
        return {
            "input_ids": __import__("torch").tensor(rows),
            "attention_mask": __import__("torch").tensor([[item != 0 for item in row] for row in rows]),
        }


def test_tokenizer_uses_explicit_bos_and_right_padding():
    tokenizer = PretrainedTextTokenizer(_Tokenizer())
    ids, mask = tokenizer.batch_encode(["a", "abc"], max_length=4)
    assert tokenizer.tokenizer.padding_side == "right"
    assert ids.tolist() == [[1, 3, 0, 0], [1, 3, 4, 5]]
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]
