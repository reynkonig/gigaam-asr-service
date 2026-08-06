import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from gigaam_asr_service.api import create_app
from gigaam_asr_service.config import MODEL_ID, MODEL_REVISION, MODEL_SOURCE
from gigaam_asr_service.engine import (
    AsrServiceError,
    AsrValidationError,
    TranscriptionResult,
)
from gigaam_asr_service.pool import AsrBusyError
from tests.helpers import make_settings


class FakeBackend:
    def __init__(self) -> None:
        self.is_loaded = False
        self.load_calls = 0
        self.close_calls = 0
        self.paths: list[str] = []
        self.payloads: list[bytes] = []
        self.error: Exception | None = None
        self.result = TranscriptionResult("тестовая расшифровка", 1.25)
        self.snapshot_values: dict[str, object] = {
            "mode": "fake",
            "worker_count": 1,
            "ready_workers": 0,
            "failed_workers": 0,
            "active_requests": 0,
            "available_workers": 1,
            "completed_requests": 0,
            "failed_requests": 0,
        }

    @property
    def loaded(self) -> bool:
        return self.is_loaded

    def load(self) -> None:
        self.load_calls += 1
        if self.error is not None:
            raise self.error
        self.is_loaded = True
        self.snapshot_values["ready_workers"] = 1

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        self.paths.append(audio_path)
        self.payloads.append(Path(audio_path).read_bytes())
        if self.error is not None:
            raise self.error
        self.is_loaded = True
        self.snapshot_values["ready_workers"] = 1
        return self.result

    def snapshot(self) -> dict[str, object]:
        return dict(self.snapshot_values)

    def close(self) -> None:
        self.close_calls += 1


def audio_file(content: bytes = b"audio") -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("sample.wav", content, "audio/wav")}


class TranscriptionApiTest(unittest.TestCase):
    def test_json_contract_uses_fixed_model_and_removes_temporary_file(self) -> None:
        backend = FakeBackend()
        with TestClient(create_app(make_settings(), backend=backend)) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": MODEL_ID, "language": "ru"},
                files=audio_file(b"wave-data"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "тестовая расшифровка"})
        self.assertEqual(response.headers["x-asr-model"], MODEL_ID)
        self.assertEqual(response.headers["x-asr-model-revision"], MODEL_REVISION)
        self.assertEqual(response.headers["x-audio-duration-seconds"], "1.250")
        self.assertEqual(backend.payloads, [b"wave-data"])
        self.assertFalse(os.path.exists(backend.paths[0]))
        self.assertEqual(backend.close_calls, 1)

    def test_text_response_and_source_model_alias_are_supported(self) -> None:
        backend = FakeBackend()
        with TestClient(create_app(make_settings(), backend=backend)) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": MODEL_SOURCE, "response_format": "text"},
                files=audio_file(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "тестовая расшифровка")
        self.assertTrue(response.headers["content-type"].startswith("text/plain"))

    def test_unsupported_options_are_rejected_before_inference(self) -> None:
        cases = [
            ({"model": "other-model"}, 400),
            ({"language": "en"}, 400),
            ({"prompt": "context"}, 400),
            ({"temperature": "0.5"}, 400),
            ({"timestamp_granularities[]": "word"}, 400),
            ({"response_format": "verbose_json"}, 422),
        ]
        for data, expected_status in cases:
            with self.subTest(data=data):
                backend = FakeBackend()
                with TestClient(create_app(make_settings(), backend=backend)) as client:
                    response = client.post(
                        "/v1/audio/transcriptions",
                        data=data,
                        files=audio_file(),
                    )
                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(backend.paths, [])

    def test_empty_and_oversized_uploads_are_rejected_and_cleaned_up(self) -> None:
        backend = FakeBackend()
        settings = make_settings(max_upload_bytes=3)
        with TestClient(create_app(settings, backend=backend)) as client:
            empty = client.post(
                "/v1/audio/transcriptions",
                files=audio_file(b""),
            )
            oversized = client.post(
                "/v1/audio/transcriptions",
                files=audio_file(b"four"),
            )
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(backend.paths, [])

    def test_backend_failures_map_to_retryable_or_validation_statuses(self) -> None:
        cases = [
            (AsrBusyError("queue full"), 429),
            (AsrValidationError("bad audio"), 422),
            (AsrServiceError("GPU failed"), 503),
        ]
        for error, expected_status in cases:
            with self.subTest(error=error):
                backend = FakeBackend()
                backend.error = error
                with TestClient(create_app(make_settings(), backend=backend)) as client:
                    response = client.post(
                        "/v1/audio/transcriptions",
                        files=audio_file(),
                    )
                self.assertEqual(response.status_code, expected_status)
                self.assertFalse(os.path.exists(backend.paths[0]))
                if expected_status == 429:
                    self.assertEqual(response.headers["retry-after"], "1")


class OperationsApiTest(unittest.TestCase):
    def test_health_and_readiness_do_not_load_the_model(self) -> None:
        backend = FakeBackend()
        with TestClient(create_app(make_settings(), backend=backend)) as client:
            health = client.get("/healthz")
            cold = client.get("/readyz")
            backend.snapshot_values.update(
                {"ready_workers": 1, "failed_workers": 1, "worker_count": 2}
            )
            degraded = client.get("/readyz")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["model"], MODEL_ID)
        self.assertEqual(health.json()["source"], MODEL_SOURCE)
        self.assertEqual(health.json()["dtype"], "float32")
        self.assertEqual(cold.status_code, 503)
        self.assertEqual(degraded.status_code, 200)
        self.assertEqual(degraded.json()["failed_workers"], 1)
        self.assertEqual(backend.load_calls, 0)

    def test_models_reports_only_gigaam(self) -> None:
        backend = FakeBackend()
        with TestClient(create_app(make_settings(), backend=backend)) as client:
            response = client.get("/v1/models")
        self.assertEqual(
            response.json(),
            {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "ai-sage"}],
            },
        )

    def test_bearer_auth_protects_models_inference_and_warmup(self) -> None:
        backend = FakeBackend()
        settings = make_settings(api_key="secret")
        with TestClient(create_app(settings, backend=backend)) as client:
            health = client.get("/healthz")
            missing = client.post(
                "/v1/audio/transcriptions",
                files=audio_file(),
            )
            wrong = client.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrong"},
            )
            warm = client.post(
                "/admin/warmup",
                headers={"Authorization": "Bearer secret"},
            )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(warm.status_code, 200)
        self.assertEqual(backend.load_calls, 1)

    def test_preload_and_shutdown_use_application_lifespan(self) -> None:
        backend = FakeBackend()
        settings = make_settings(preload=True)
        with TestClient(create_app(settings, backend=backend)) as client:
            self.assertEqual(backend.load_calls, 1)
            self.assertEqual(client.get("/readyz").status_code, 200)
            self.assertEqual(backend.close_calls, 0)
        self.assertEqual(backend.close_calls, 1)

    def test_warmup_failure_is_unavailable(self) -> None:
        backend = FakeBackend()
        backend.error = AsrServiceError("cannot load")
        with TestClient(create_app(make_settings(), backend=backend)) as client:
            response = client.post("/admin/warmup")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "cannot load")


if __name__ == "__main__":
    unittest.main()
