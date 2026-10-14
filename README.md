# GigaAM ASR service

OpenAI-compatible transcription service for one fixed model:
`gigaam-v3-e2e-rnnt`, backed by `ai-sage/GigaAM-v3` at the pinned revision
`7655ad717f8122257385bb4b2f373db3697e8680`. Inference uses FP32 ONNX Runtime;
PyTorch is only needed for the one-time model export.

The service returns Russian text only. It does not run VAD, speaker
diarization, or timestamp generation. Uploads must be short chunks of at most
24 seconds. Stereo audio is downmixed by the model, so callers that need speaker
roles must split channels before sending requests.

## Docker

```bash
cp .env.example .env
mkdir -p .local/cache/huggingface
docker compose run --build --rm export-model
docker compose up --build -d gigaam-asr-service
docker compose logs -f gigaam-asr-service
```

The exporter initializes the cache as container root, including with rootless
Docker. The serving process runs as the non-root `app` user.
The API listens on `127.0.0.1:8090` by default. Model files are stored in
`.local/cache/huggingface`, mounted at `/cache/huggingface`. The exporter downloads
the pinned weights, checks the NumPy frontend and all ONNX networks against PyTorch,
and writes the ONNX graphs, tokenizer, and frontend constants to
`$HF_HOME/onnx/7655ad717f8122257385bb4b2f373db3697e8680`. Repeating the export
reuses the completed artifacts. Keep this cache when replacing the serving image.

The serving image contains ONNX Runtime and its CUDA/cuDNN libraries, with no
PyTorch, Transformers, or Triton. The encoder uses the selected device; the small
prediction networks run on CPU. Startup requires the exported files and does
not download models. `ASR_LOCAL_FILES_ONLY` remains accepted; inference always
reads local artifacts. To export offline from cached Hugging Face weights, append
`--local-files-only` to the export command. The exporter image can be built on a
machine without a GPU and is not needed on the serving host after copying the
exported directory to the same cache location.

One model replica per GPU:

```dotenv
ASR_DEVICES=cuda:0,cuda:1
```

Multiple replicas can share a GPU when it has enough memory:

```dotenv
ASR_DEVICES=cuda:0
ASR_REPLICAS_PER_DEVICE=2
```

Replicas increase concurrent request throughput and each replica loads another
full model copy. Keep one Uvicorn worker because every Uvicorn worker would
create another model pool. Docker GPU access requires the NVIDIA Container
Toolkit.

## API

```bash
curl http://127.0.0.1:8090/healthz
curl http://127.0.0.1:8090/readyz

curl http://127.0.0.1:8090/v1/audio/transcriptions \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F model=gigaam-v3-e2e-rnnt \
  -F language=ru \
  -F response_format=json \
  -F file=@chunk.wav

curl -X POST http://127.0.0.1:8090/admin/warmup \
  -H "Authorization: Bearer $ASR_API_KEY"
```

`response_format` supports `json` and `text`. Prompts, nonzero temperature,
timestamp granularities, and languages other than Russian are rejected because
the model does not support them.

`GET /v1/models` returns the single available model and requires authentication.
Health checks do not load the model. Readiness returns HTTP 503 until at least
one replica is loaded.

## Local development

Python 3.13, `uv`, `ffmpeg`, and `ffprobe` are required.

```bash
uv sync
uv run python -m unittest
uv run ruff check .
uv run --group export python -m gigaam_asr_service.export_onnx
uv run gigaam-asr-server --devices cpu --in-process
```

For local GPU inference, install the CUDA libraries with `uv sync --group cuda`
and use `uv run --group cuda gigaam-asr-server --devices cuda:0`.

Use TLS or an SSH tunnel when exposing the service over a network.
