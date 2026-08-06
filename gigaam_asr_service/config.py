from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

MODEL_ID = "gigaam-v3-e2e-rnnt"
MODEL_SOURCE = "ai-sage/GigaAM-v3"
MODEL_REVISION = "7655ad717f8122257385bb4b2f373db3697e8680"
MODEL_ALIASES = frozenset({MODEL_ID, MODEL_SOURCE})
MODEL_LANGUAGE = "ru"
MODEL_MAX_AUDIO_SECONDS = 24.9

DEVICE_PATTERN = re.compile(r"(?:auto|cpu|cuda(?::\d+)?)\Z")


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _boolean(name: str, default: bool) -> bool:
    value = _optional_text(os.getenv(name))
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _integer(name: str, default: int) -> int:
    value = _optional_text(os.getenv(name))
    return default if value is None else int(value)


def _float(name: str, default: float) -> float:
    value = _optional_text(os.getenv(name))
    return default if value is None else float(value)


def _devices(value: str) -> tuple[str, ...]:
    devices = tuple(
        "cuda:0" if item.strip() == "cuda" else item.strip()
        for item in value.split(",")
        if item.strip()
    )
    if not devices:
        raise ValueError("ASR_DEVICES must contain at least one device")
    return devices


@dataclass(frozen=True)
class ServerSettings:
    devices: tuple[str, ...] = ("auto",)
    local_files_only: bool = False
    max_upload_bytes: int = 25 * 1024 * 1024
    max_audio_seconds: float = 24.0
    max_pending_requests: int = 32
    queue_timeout_seconds: float = 30.0
    inference_timeout_seconds: float = 300.0
    worker_restart_cooldown_seconds: float = 30.0
    shutdown_timeout_seconds: float = 30.0
    ffprobe_timeout_seconds: float = 15.0
    preload: bool = False
    api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 8_090

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("devices must contain at least one device")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices must not contain duplicates")
        if "auto" in self.devices and len(self.devices) != 1:
            raise ValueError("auto cannot be combined with explicit devices")
        for device in self.devices:
            if DEVICE_PATTERN.fullmatch(device) is None:
                raise ValueError(
                    "devices must contain only auto, cpu, or cuda device indexes"
                )
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be positive")
        if not math.isfinite(self.max_audio_seconds) or not (
            0 < self.max_audio_seconds <= MODEL_MAX_AUDIO_SECONDS
        ):
            raise ValueError(
                f"max_audio_seconds must be between 0 and {MODEL_MAX_AUDIO_SECONDS}"
            )
        if self.max_pending_requests < 0:
            raise ValueError("max_pending_requests must not be negative")
        for name in (
            "queue_timeout_seconds",
            "inference_timeout_seconds",
            "worker_restart_cooldown_seconds",
            "shutdown_timeout_seconds",
            "ffprobe_timeout_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.api_key:
            raise ValueError("ASR_API_KEY is required when binding beyond loopback")

    @classmethod
    def from_env(cls, **overrides: object) -> ServerSettings:
        devices_value = (
            _optional_text(os.getenv("ASR_DEVICES"))
            or _optional_text(os.getenv("ASR_DEVICE"))
            or "auto"
        )
        values: dict[str, object] = {
            "devices": _devices(devices_value),
            "local_files_only": _boolean("ASR_LOCAL_FILES_ONLY", False),
            "max_upload_bytes": _integer("ASR_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
            "max_audio_seconds": _float("ASR_MAX_AUDIO_SECONDS", 24.0),
            "max_pending_requests": _integer("ASR_MAX_PENDING_REQUESTS", 32),
            "queue_timeout_seconds": _float("ASR_QUEUE_TIMEOUT_SECONDS", 30.0),
            "inference_timeout_seconds": _float("ASR_INFERENCE_TIMEOUT_SECONDS", 300.0),
            "worker_restart_cooldown_seconds": _float(
                "ASR_WORKER_RESTART_COOLDOWN_SECONDS", 30.0
            ),
            "shutdown_timeout_seconds": _float("ASR_SHUTDOWN_TIMEOUT_SECONDS", 30.0),
            "ffprobe_timeout_seconds": _float("ASR_FFPROBE_TIMEOUT_SECONDS", 15.0),
            "preload": _boolean("ASR_PRELOAD", False),
            "api_key": _optional_text(os.getenv("ASR_API_KEY")),
            "host": _optional_text(os.getenv("ASR_HOST")) or "127.0.0.1",
            "port": _integer("ASR_PORT", 8_090),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]
