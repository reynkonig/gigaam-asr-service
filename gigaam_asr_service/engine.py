from __future__ import annotations

import logging
import threading
import wave
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

    def _load_audio(self, audio_path: str) -> np.ndarray:
        try:
            with wave.open(audio_path, "rb") as audio:
                if (
                    audio.getnchannels(),
                    audio.getframerate(),
                    audio.getsampwidth(),
                    audio.getcomptype(),
                ) != (1, 16000, 2, "NONE"):
                    raise AsrValidationError(
                        "audio must be a mono 16 kHz 16-bit PCM WAV file"
                    )
                frames = audio.getnframes()
                if frames < 320:
                    raise AsrValidationError(
                        "audio must contain at least 20 ms of samples"
                    )
                if frames > self.settings.max_audio_seconds * 16000:
                    raise AsrValidationError(
                        "audio duration exceeds the configured limit of "
                        f"{self.settings.max_audio_seconds:g} seconds"
                    )
                pcm = audio.readframes(frames + 1)
                if len(pcm) != frames * 2:
                    raise AsrValidationError(
                        "WAV data does not match its declared frame count"
                    )
        except AsrValidationError:
            raise
        except (OSError, EOFError, RuntimeError, wave.Error) as error:
            raise AsrValidationError(
                "audio must be a readable mono 16 kHz 16-bit PCM WAV file"
            ) from error
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        audio = self._load_audio(str(audio_path))
        self.load()
        assert self._model is not None
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
        return TranscriptionResult(
            text=text.strip(), duration_seconds=audio.size / 16000
        )
