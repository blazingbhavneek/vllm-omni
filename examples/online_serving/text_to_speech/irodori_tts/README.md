# Irodori-TTS v4-Small server

Start the server with `./run_server.sh`, then use `speech_client.py`. The launch
script enables step batching for up to four active requests and enables
Irodori CUDA graphs. Override the capacity with `MAX_NUM_SEQS`, or compare with
graphs disabled by setting `IRODORI_CUDA_GRAPH=0`:

```bash
MAX_NUM_SEQS=4 ./run_server.sh
IRODORI_CUDA_GRAPH=0 MAX_NUM_SEQS=4 ./run_server.sh
```

Send two same-bucket requests at the same time to exercise one fused denoise
batch and one CUDA graph key:

```bash
python speech_client.py --text '一つ目の音声です。' --seconds 4 --output first.wav &
python speech_client.py --text '二つ目の音声です。' --seconds 4 --output second.wav &
wait
```

Change the second request to `--seconds 8` to check that mixed buckets stay
active together but run as separate physical microbatches.

The API
accepts `instructions` for caption conditioning and an ordered JSON `ref_audio`
list for voice cloning. References are resolved by the server; `file://` URIs
still require the server's allowed-local-media policy. Responses are final-only
WAV or PCM, mono 48 kHz, and are not SilentCipher-watermarked in this release.
