#!/usr/bin/env bash
set -euo pipefail


vllm serve "${MODEL:-/mnt/common/Code/tts-models-support/Aratako/Irodori-TTS-v4-Small}" \
  --host 0.0.0.0 \
  --port "${PORT:-8091}" \
  --dtype "${DTYPE:-float32}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --step-execution \
  --omni
