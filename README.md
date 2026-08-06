# GigaAM ASR service

OpenAI-compatible transcription service for one fixed model:
`gigaam-v3-e2e-rnnt`, backed by `ai-sage/GigaAM-v3` at the pinned revision
`7655ad717f8122257385bb4b2f373db3697e8680`.

The service returns Russian text only. It does not run VAD, speaker
diarization, or timestamp generation. Uploads must be short chunks of at most
24 seconds. Stereo audio is downmixed by the model, so callers that need speaker
roles must split channels before sending requests.

## Docker

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f gigaam-asr-service
```

The API listens on `127.0.0.1:8090` by default. Model files are stored in the
`gigaam-asr-hf-cache` Docker volume.

One model replica per GPU:

```dotenv
ASR_DEVICES=cuda:0,cuda:1
```

Replicas increase request throughput. Keep one Uvicorn worker because every
Uvicorn worker would create another model pool. Docker GPU access requires the
NVIDIA Container Toolkit.

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
uv run gigaam-asr-server --devices cpu --in-process
```

Use TLS or an SSH tunnel when exposing the service over a network.
