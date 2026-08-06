from gigaam_asr_service.config import ServerSettings


def make_settings(**overrides: object) -> ServerSettings:
    values: dict[str, object] = {
        "devices": ("cpu",),
        "max_upload_bytes": 1024,
        "max_audio_seconds": 24.0,
        "max_pending_requests": 2,
        "queue_timeout_seconds": 0.2,
        "inference_timeout_seconds": 1.0,
        "worker_restart_cooldown_seconds": 0.1,
        "shutdown_timeout_seconds": 0.1,
        "ffprobe_timeout_seconds": 1.0,
        "host": "127.0.0.1",
        "port": 8090,
    }
    values.update(overrides)
    return ServerSettings(**values)  # type: ignore[arg-type]
