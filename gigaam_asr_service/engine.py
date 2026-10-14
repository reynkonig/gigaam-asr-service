from __future__ import annotations

import logging
import math
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ServerSettings
from .onnx_model import OnnxModel, model_directory

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

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
                    raise AsrServiceError("ffmpeg and ffprobe are required")

                model = OnnxModel(model_directory(), self.requested_device)
            except AsrServiceError as error:
                self._last_error = type(error).__name__
                raise
            except Exception as error:
                self._last_error = type(error).__name__
                LOGGER.exception("Failed to load GigaAM on %s", self.requested_device)
                raise AsrServiceError(
                    f"failed to load GigaAM on {self.requested_device}: {error}"
                ) from error

            self._model = model
            self._resolved_device = model.resolved_device
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

    def _load_audio(self, audio_path: str) -> np.ndarray:
        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    audio_path,
                    "-t",
                    str(self.settings.max_audio_seconds + 0.1),
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-",
                ],
                capture_output=True,
                check=True,
                timeout=self.settings.inference_timeout_seconds,
            )
            audio = (
                np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32)
                / 32768.0
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise AsrValidationError("audio file could not be decoded") from error
        if audio.size < 320:
            raise AsrValidationError("audio must contain at least 20 ms of samples")
        if audio.size > self.settings.max_audio_seconds * 16000:
            raise AsrValidationError(
                "decoded audio exceeds the configured duration limit"
            )
        return audio

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        duration = self._duration_seconds(str(audio_path))
        self.load()
        assert self._model is not None
        audio = self._load_audio(str(audio_path))
        try:
            with self._inference_lock:
                text = self._model.transcribe(audio)
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
