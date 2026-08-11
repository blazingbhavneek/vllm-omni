#!/usr/bin/env bash
set -euo pipefail

vllm serve "${MODEL:-Aratako/Irodori-TTS-v4-Small}" \
  --host 0.0.0.0 \
  --port "${PORT:-8091}" \
  --omni
