# Irodori-TTS v4-Small

Install the optional codec bridge first:

```bash
uv pip install -e '.[irodori-tts]'
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
