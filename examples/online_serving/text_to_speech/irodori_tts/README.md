# Irodori-TTS v4-Small server

Start the server with `./run_server.sh`, then use `speech_client.py`. The API
accepts `instructions` for caption conditioning and an ordered JSON `ref_audio`
list for voice cloning. References are resolved by the server; `file://` URIs
still require the server's allowed-local-media policy. Responses are final-only
WAV or PCM, mono 48 kHz, and are not SilentCipher-watermarked in this release.
