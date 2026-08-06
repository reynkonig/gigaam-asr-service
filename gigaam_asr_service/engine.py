from __future__ import annotations

import logging
import math
import shutil
import subprocess
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MODEL_REVISION, MODEL_SOURCE, ServerSettings

LOGGER = logging.getLogger(__name__)


class AsrServiceError(RuntimeError):
    pass


class AsrValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration_seconds: float


class GigaAmEngine:
    def __init__(self, settings: ServerSettings, device: str) -> None:
        self.settings = settings
        self.requested_device = device
        self._model: Any = None
        self._torch: Any = None
        self._resolved_device: str | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def snapshot(self) -> dict[str, object]:
        return {
            "device": self.requested_device,
            "resolved_device": self._resolved_device,
            "loaded": self.loaded,
            "state": "ready"
            if self.loaded
            else ("failed" if self._last_error else "cold"),
            "last_error": self._last_error,
            "dtype": "float32",
        }

    def _resolve_device(self, torch: Any) -> Any:
        requested = self.requested_device
        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise AsrServiceError(
                    f"CUDA device {requested!r} was requested but CUDA is unavailable"
                )
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                raise AsrServiceError(
                    f"CUDA device {requested!r} does not exist; "
                    f"visible device count is {torch.cuda.device_count()}"
                )
        return device

    @staticmethod
    def _cuda_fp32_forward(
        model: Any,
        features: Any,
        feature_lengths: Any,
    ) -> tuple[Any, Any]:
        features, feature_lengths = model.preprocessor(features, feature_lengths)
        return model.encoder(features.float(), feature_lengths)

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
                    raise AsrServiceError("ffmpeg and ffprobe are required")

                import torch
                from transformers import AutoModel

                device = self._resolve_device(torch)
                model = AutoModel.from_pretrained(
                    MODEL_SOURCE,
                    revision=MODEL_REVISION,
                    trust_remote_code=True,
                    local_files_only=self.settings.local_files_only,
                )
                if device.type == "cuda":
                    model.model.forward = types.MethodType(
                        self._cuda_fp32_forward,
                        model.model,
                    )
                model = model.float().to(device).eval()
            except AsrServiceError as error:
                self._last_error = type(error).__name__
                raise
            except Exception as error:
                self._last_error = type(error).__name__
                LOGGER.exception("Failed to load GigaAM on %s", self.requested_device)
                raise AsrServiceError(
                    f"failed to load GigaAM on {self.requested_device}"
                ) from error

            self._torch = torch
            self._model = model
            self._resolved_device = str(device)
            self._last_error = None

    def _duration_seconds(self, audio_path: str) -> float:
        try:
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=self.settings.ffprobe_timeout_seconds,
            )
            duration = float(completed.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise AsrValidationError("audio file could not be decoded") from error
        if not math.isfinite(duration) or duration <= 0:
            raise AsrValidationError("audio duration must be positive")
        if duration > self.settings.max_audio_seconds:
            raise AsrValidationError(
                "audio duration exceeds the configured limit of "
                f"{self.settings.max_audio_seconds:g} seconds"
            )
        return duration

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        duration = self._duration_seconds(str(audio_path))
        self.load()
        assert self._model is not None
        assert self._torch is not None
        try:
            with self._inference_lock, self._torch.inference_mode():
                text = self._model.transcribe(str(audio_path))
        except ValueError as error:
            if "too long" in str(error).lower():
                raise AsrValidationError(
                    "audio is too long for the GigaAM short-form decoder"
                ) from error
            self._last_error = type(error).__name__
            raise AsrServiceError(
                f"GigaAM inference failed on {self.requested_device}"
            ) from error
        except Exception as error:
            self._last_error = type(error).__name__
            LOGGER.exception("GigaAM inference failed on %s", self.requested_device)
            raise AsrServiceError(
                f"GigaAM inference failed on {self.requested_device}"
            ) from error
        if not isinstance(text, str):
            self._last_error = "InvalidModelOutput"
            raise AsrServiceError("GigaAM returned an invalid transcription")
        self._last_error = None
        return TranscriptionResult(text=text.strip(), duration_seconds=duration)
