# Irodori-TTS v4-Small

Install the optional codec bridge first:

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" -e '.[irodori-tts]'
```

Text only:

```bash
python end2end.py --text 'こんにちは。これはテストです。'
```

Caption-controlled synthesis:

```bash
python end2end.py --caption '落ち着いた、やさしい話し方'
```

Ordered multi-reference voice cloning:

```bash
python end2end.py --ref-audio ref1.wav --ref-audio ref2.wav
```

The model is final-only, produces mono 48 kHz audio, and the first native
release does not apply a SilentCipher watermark.

## Local checkpoint / step-batch smoke test

`tester.py` defaults to the same long reference-cloning prompt, caption, and
`clone_ref1.wav` from the second command in the workspace's `command.txt`.
Use the same fixed seed in both runners when comparing output:

```bash
# From Irodori-TTS/
uv run --no-sync python infer.py \
  --checkpoint ../Aratako/Irodori-TTS-v4-Small/model.safetensors \
  --ref-wav ../Aratako/Irodori-TTS-v4-Small/samples/clone_ref1.wav \
  --text 'ねえ、今日の帰りに少しだけ寄り道しない？🤔 新しくできた小さな喫茶店があるらしいんだ。⏸️ 🫶 急がなくていいから、あたたかいコーヒーを飲みながら、ゆっくり話せたらうれしいな。' \
  --caption '親しい相手にやさしく語りかける、穏やかで少し照れくさい大人の男性。自然な会話調で、柔らかく温度感のある声。' \
  --seed 0 --output-wav outputs/upstream.wav

# From vllm-omni/
uv run --active --no-sync python examples/offline_inference/text_to_speech/irodori_tts/tester.py \
  --seed 0 --output-wav outputs/irodori-vllm.wav
```

The tester resolves its default local checkpoint and reference relative to the
workspace, so its defaults work from any current directory. Override the
inputs with `--checkpoint`, `--text`, `--caption`, and `--ref-wav`; add
`--no-ref` for the text-only command shape.

```bash
uv run --active --no-sync python tester.py \
  --checkpoint ../../../../../Aratako/Irodori-TTS-v4-Small/model.safetensors \
  --text 'こんにちは。これはGPUで生成した通常品質の音声合成テストです。' \
  --caption '落ち着いた自然な日本語の話し声で、明瞭に読み上げる。' \
  --no-ref --model-precision bf16 --codec-precision bf16 \
  --output-wav outputs/irodori-vllm.wav
```

It enables step execution by default. To exercise predicted-duration
continuous batching, pass more than one `--text` and set the active capacity:

```bash
uv run --active --no-sync python tester.py \
  --checkpoint ../../../../../Aratako/Irodori-TTS-v4-Small/model.safetensors \
  --text '一つ目のリクエストです。' --text '二つ目のリクエストです。' \
  --max-num-seqs 2 --num-steps 4
```

Add `--seconds 3` when both requests should use the same explicit duration.
CUDA graphs are also enabled by default. A shape runs eagerly once, is captured
on its second denoise step, and is replayed after that. Use
`--disable-cuda-graph` to compare against step-batched eager execution, or
`--enforce-eager` to force the engine's eager mode.
