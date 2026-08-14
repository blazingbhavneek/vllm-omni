# Irodori-TTS test setup

The checked-in contract tests do not require the official Irodori server:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/diffusion/models/test_irodori_tts.py \
  tests/entrypoints/openai_api/test_tts_adapter.py \
  tests/entrypoints/openai_api/test_tts_detection.py
```

For a local vLLM-Omni model smoke test, install the runtime extra in the
vLLM-Omni environment:

```bash
pip install -e ".[irodori-tts]"
python examples/offline_inference/text_to_speech/irodori_tts/end2end.py \
  --text "こんにちは。これは音声合成のテストです。" \
  --seconds 4
```

## Comparing with the official implementation

Install `Irodori-TTS-Server` in a separate virtual environment. Its PyTorch
requirements are independent of vLLM-Omni's and must not be added to the
vLLM-Omni dependency set.

```bash
git clone https://github.com/Aratako/Irodori-TTS-Server.git
cd Irodori-TTS-Server
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Launch the official FP32 reference without a local checkpoint path:

```bash
env \
  IRODORI_HOST=127.0.0.1 \
  IRODORI_PORT=8088 \
  IRODORI_HF_CHECKPOINT=Aratako/Irodori-TTS-v4-Small \
  IRODORI_MODEL_DEVICE=cuda \
  IRODORI_CODEC_DEVICE=cuda \
  IRODORI_MODEL_PRECISION=fp32 \
  IRODORI_CODEC_PRECISION=fp32 \
  IRODORI_PRELOAD=true \
  IRODORI_MAX_CONCURRENT_SYNTHESIS=1 \
  python -m irodori_openai_tts
```

Use the same text, caption, seed, number of denoising steps, explicit output
duration, and reference audio for both implementations. Compare decoded 48 kHz
waveforms with identical sample counts. The official FP32 output is the parity
reference; BF16 outputs should not be used as the baseline.
