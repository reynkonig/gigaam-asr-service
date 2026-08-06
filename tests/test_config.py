import os
import unittest
from unittest.mock import patch

from gigaam_asr_service.config import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SOURCE,
    ServerSettings,
)
from tests.helpers import make_settings


class ServerSettingsTest(unittest.TestCase):
    def test_model_identity_is_fixed_and_revision_is_pinned(self) -> None:
        self.assertEqual(MODEL_ID, "gigaam-v3-e2e-rnnt")
        self.assertEqual(MODEL_SOURCE, "ai-sage/GigaAM-v3")
        self.assertRegex(MODEL_REVISION, r"^[0-9a-f]{40}$")

    def test_environment_builds_replica_settings(self) -> None:
        environment = {
            "ASR_DEVICES": "cuda,cuda:2",
            "ASR_MAX_UPLOAD_BYTES": "4096",
            "ASR_MAX_AUDIO_SECONDS": "22",
            "ASR_MAX_PENDING_REQUESTS": "7",
            "ASR_PRELOAD": "true",
            "ASR_HOST": "127.0.0.1",
            "ASR_PORT": "9000",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = ServerSettings.from_env()

        self.assertEqual(settings.devices, ("cuda:0", "cuda:2"))
        self.assertEqual(settings.max_upload_bytes, 4096)
        self.assertEqual(settings.max_audio_seconds, 22)
        self.assertEqual(settings.max_pending_requests, 7)
        self.assertTrue(settings.preload)
        self.assertEqual(settings.port, 9000)

    def test_environment_device_alias_is_supported(self) -> None:
        with patch.dict(os.environ, {"ASR_DEVICE": "cpu"}, clear=True):
            settings = ServerSettings.from_env()
        self.assertEqual(settings.devices, ("cpu",))

    def test_invalid_devices_are_rejected(self) -> None:
        cases = [
            (("auto", "cuda:0"), "auto"),
            (("cuda:0", "cuda:0"), "duplicates"),
            (("mps",), "only auto"),
        ]
        for devices, message in cases:
            with (
                self.subTest(devices=devices),
                self.assertRaisesRegex(ValueError, message),
            ):
                make_settings(devices=devices)

    def test_limits_and_timeouts_are_validated(self) -> None:
        cases = [
            ({"max_upload_bytes": 0}, "max_upload_bytes"),
            ({"max_audio_seconds": 25}, "max_audio_seconds"),
            ({"max_pending_requests": -1}, "max_pending_requests"),
            ({"queue_timeout_seconds": 0}, "queue_timeout_seconds"),
            ({"port": 0}, "port"),
        ]
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                make_settings(**overrides)

    def test_non_loopback_binding_requires_authentication(self) -> None:
        with self.assertRaisesRegex(ValueError, "ASR_API_KEY"):
            make_settings(host="0.0.0.0")
        settings = make_settings(host="0.0.0.0", api_key="secret")
        self.assertEqual(settings.api_key, "secret")

    def test_boolean_environment_values_are_strict(self) -> None:
        with (
            patch.dict(os.environ, {"ASR_PRELOAD": "maybe"}, clear=True),
            self.assertRaisesRegex(ValueError, "true or false"),
        ):
            ServerSettings.from_env()


if __name__ == "__main__":
    unittest.main()
