from __future__ import annotations

import asyncio
import hmac
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from .config import (
    MODEL_ALIASES,
    MODEL_ID,
    MODEL_LANGUAGE,
    MODEL_REVISION,
    MODEL_SOURCE,
    ServerSettings,
)
from .engine import AsrServiceError, AsrValidationError
from .pool import AsrBackend, AsrBusyError, AsrProcessPool

UPLOAD_CHUNK_BYTES = 1024 * 1024
SAFE_SUFFIX_PATTERN = re.compile(r"\.[a-zA-Z0-9]{1,10}\Z")


class UploadTooLargeError(ValueError):
    pass


class ModelData(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelData]


def _save_upload(upload: UploadFile, max_bytes: int) -> Path:
    suffix = Path(upload.filename or "").suffix
    if SAFE_SUFFIX_PATTERN.fullmatch(suffix) is None:
        suffix = ".audio"
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="gigaam-asr-",
            suffix=suffix.lower(),
            delete=False,
        ) as temporary:
            path = Path(temporary.name)
            total = 0
            while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise UploadTooLargeError(
                        f"audio upload exceeds the {max_bytes}-byte limit"
                    )
                temporary.write(chunk)
        if total == 0:
            raise ValueError("audio upload must not be empty")
        return path
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def _backend_snapshot(backend: AsrBackend) -> dict[str, object]:
    snapshot = backend.snapshot()
    return snapshot if isinstance(snapshot, dict) else {}


def create_app(
    settings: ServerSettings | None = None,
    *,
    backend: AsrBackend | None = None,
) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    backend = backend or AsrProcessPool(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if settings.preload:
                await asyncio.to_thread(backend.load)
            yield
        finally:
            await asyncio.to_thread(backend.close)

    app = FastAPI(
        title="GigaAM ASR Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.asr_backend = backend
    app.state.asr_settings = settings

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        if settings.api_key is None:
            return
        expected = f"Bearer {settings.api_key}"
        if authorization is None or not hmac.compare_digest(
            authorization,
            expected,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/healthz")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model": MODEL_ID,
            "source": MODEL_SOURCE,
            "revision": MODEL_REVISION,
            "language": MODEL_LANGUAGE,
            "dtype": "float32",
            "max_audio_seconds": settings.max_audio_seconds,
            "loaded": backend.loaded,
            **_backend_snapshot(backend),
        }

    @app.get("/readyz")
    def readiness() -> JSONResponse:
        snapshot = _backend_snapshot(backend)
        ready_workers = int(snapshot.get("ready_workers", int(backend.loaded)))
        failed_workers = int(snapshot.get("failed_workers", 0))
        worker_count = int(snapshot.get("worker_count", 1))
        ready = ready_workers > 0
        return JSONResponse(
            {
                "status": "ok" if ready else "not_ready",
                "ready": ready,
                "ready_workers": ready_workers,
                "failed_workers": failed_workers,
                "worker_count": worker_count,
            },
            status_code=(
                status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
        )

    @app.post("/admin/warmup", dependencies=[Depends(authenticate)])
    def warmup() -> dict[str, object]:
        try:
            backend.load()
        except AsrServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        return {"status": "ok", **_backend_snapshot(backend)}

    @app.get(
        "/v1/models",
        response_model=ModelList,
        dependencies=[Depends(authenticate)],
    )
    def models() -> ModelList:
        return ModelList(data=[ModelData(id=MODEL_ID, owned_by="ai-sage")])

    @app.post(
        "/v1/audio/transcriptions",
        dependencies=[Depends(authenticate)],
        response_model=None,
    )
    def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = MODEL_ID,
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        response_format: Annotated[Literal["json", "text"], Form()] = "json",
        temperature: Annotated[float, Form()] = 0.0,
        timestamp_granularities: Annotated[
            list[str] | None,
            Form(alias="timestamp_granularities[]"),
        ] = None,
    ) -> Response:
        if model not in MODEL_ALIASES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only model {MODEL_ID!r} is available",
            )
        if language is not None and language.strip().lower() != MODEL_LANGUAGE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GigaAM supports only language='ru'",
            )
        if prompt is not None and prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GigaAM does not support transcription prompts",
            )
        if temperature != 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GigaAM supports only temperature=0",
            )
        if timestamp_granularities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GigaAM does not provide timestamp granularities",
            )

        try:
            audio_path = _save_upload(file, settings.max_upload_bytes)
        except UploadTooLargeError as error:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

        try:
            result = backend.transcribe(str(audio_path))
        except AsrBusyError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
                headers={"Retry-After": "1"},
            ) from error
        except AsrValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        except AsrServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        finally:
            audio_path.unlink(missing_ok=True)

        headers = {
            "X-ASR-Model": MODEL_ID,
            "X-ASR-Model-Revision": MODEL_REVISION,
            "X-Audio-Duration-Seconds": f"{result.duration_seconds:.3f}",
        }
        if response_format == "text":
            return PlainTextResponse(result.text, headers=headers)
        return JSONResponse({"text": result.text}, headers=headers)

    return app
