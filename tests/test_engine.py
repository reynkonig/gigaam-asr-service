import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gigaam_asr_service.engine import AsrServiceError, AsrValidationError, GigaAmEngine
from tests.helpers import make_settings


class FakeModel:
    resolved_device = "cpu"
    output: object = "распознанный текст"

    def transcribe(self, audio: np.ndarray) -> object:
        return self.output


class GigaAmEngineTest(unittest.TestCase):
    def test_load_uses_onnx_artifacts_and_requested_device_once(self) -> None:
        model = FakeModel()
        model.resolved_device = "cuda:0"
        engine = GigaAmEngine(make_settings(devices=("cuda:0",)), "cuda:0")
        with (
            patch(
                "gigaam_asr_service.engine.model_directory", return_value=Path("cache")
            ),
            patch("gigaam_asr_service.engine.OnnxModel", return_value=model) as factory,
            patch("gigaam_asr_service.engine.shutil.which", return_value="/bin/tool"),
        ):
            engine.load()
            engine.load()
        factory.assert_called_once_with(Path("cache"), "cuda:0")
        self.assertEqual(engine.snapshot()["dtype"], "float32")
        self.assertEqual(engine.snapshot()["resolved_device"], "cuda:0")

    def test_load_failure_is_reported_and_can_be_retried(self) -> None:
        engine = GigaAmEngine(make_settings(), "cpu")
        with (
            patch("gigaam_asr_service.engine.shutil.which", return_value="/bin/tool"),
            patch(
                "gigaam_asr_service.engine.OnnxModel",
                side_effect=ValueError("export first"),
            ),
            self.assertLogs("gigaam_asr_service.engine", level="ERROR"),
            self.assertRaisesRegex(AsrServiceError, "export first"),
        ):
            engine.load()
        self.assertFalse(engine.loaded)
        self.assertEqual(engine.snapshot()["state"], "failed")
        with (
            patch("gigaam_asr_service.engine.shutil.which", return_value="/bin/tool"),
            patch("gigaam_asr_service.engine.OnnxModel", return_value=FakeModel()),
        ):
            engine.load()
        self.assertEqual(engine.snapshot()["state"], "ready")
        self.assertIsNone(engine.snapshot()["last_error"])

    def test_duration_probe_validates_media_and_limit(self) -> None:
        engine = GigaAmEngine(make_settings(max_audio_seconds=10), "cpu")
        with patch(
            "gigaam_asr_service.engine.subprocess.run",
            return_value=SimpleNamespace(stdout="4.25\n"),
        ) as run:
            self.assertEqual(engine._duration_seconds("sample.wav"), 4.25)
        self.assertIn("ffprobe", run.call_args.args[0][0])
        for outcome, message in (
            (SimpleNamespace(stdout="11"), "exceeds"),
            (SimpleNamespace(stdout="nan"), "positive"),
        ):
            with (
                patch("gigaam_asr_service.engine.subprocess.run", return_value=outcome),
                self.assertRaisesRegex(AsrValidationError, message),
            ):
                engine._duration_seconds("sample.wav")
        with (
            patch(
                "gigaam_asr_service.engine.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["ffprobe"]),
            ),
            self.assertRaisesRegex(AsrValidationError, "could not be decoded"),
        ):
            engine._duration_seconds("broken.wav")

    def test_decoded_audio_is_normalized_and_checked_against_limit(self) -> None:
        engine = GigaAmEngine(make_settings(max_audio_seconds=1), "cpu")
        pcm = np.full(16000, -32768, dtype="<i2").tobytes()
        with patch(
            "gigaam_asr_service.engine.subprocess.run",
            return_value=SimpleNamespace(stdout=pcm),
        ) as run:
            audio = engine._load_audio("sample.wav")
        np.testing.assert_array_equal(audio, np.full(16000, -1, dtype=np.float32))
        self.assertEqual(run.call_args.kwargs["timeout"], 1.0)
        for pcm, message in (
            (b"", "20 ms"),
            (np.zeros(16001, dtype="<i2").tobytes(), "duration limit"),
            (b"x", "could not be decoded"),
        ):
            with (
                self.subTest(message=message),
                patch(
                    "gigaam_asr_service.engine.subprocess.run",
                    return_value=SimpleNamespace(stdout=pcm),
                ),
                self.assertRaisesRegex(AsrValidationError, message),
            ):
                engine._load_audio("sample.wav")

    def test_audio_decode_failure_is_validation_error(self) -> None:
        engine = GigaAmEngine(make_settings(), "cpu")
        for error in (
            subprocess.CalledProcessError(1, ["ffmpeg"]),
            subprocess.TimeoutExpired(["ffmpeg"], 1),
        ):
            with (
                self.subTest(error=type(error)),
                patch("gigaam_asr_service.engine.subprocess.run", side_effect=error),
                self.assertRaisesRegex(AsrValidationError, "could not be decoded"),
            ):
                engine._load_audio("broken.wav")

    def test_transcription_is_trimmed_and_preserves_duration(self) -> None:
        model = FakeModel()
        model.output = "  готовый текст  "
        engine = GigaAmEngine(make_settings(), "cpu")
        engine._model = model
        with (
            patch.object(engine, "_duration_seconds", return_value=2.5),
            patch.object(
                engine, "_load_audio", return_value=np.zeros(16000, np.float32)
            ),
        ):
            result = engine.transcribe(Path("sample.wav"))
        self.assertEqual(result.text, "готовый текст")
        self.assertEqual(result.duration_seconds, 2.5)

    def test_invalid_model_output_is_service_failure(self) -> None:
        model = FakeModel()
        model.output = None
        engine = GigaAmEngine(make_settings(), "cpu")
        engine._model = model
        with (
            patch.object(engine, "_duration_seconds", return_value=1.0),
            patch.object(
                engine, "_load_audio", return_value=np.zeros(16000, np.float32)
            ),
            self.assertRaisesRegex(AsrServiceError, "invalid transcription"),
        ):
            engine.transcribe("sample.wav")

    def test_ffmpeg_is_required_before_model_loading(self) -> None:
        engine = GigaAmEngine(make_settings(), "cpu")
        with (
            patch("gigaam_asr_service.engine.shutil.which", return_value=None),
            self.assertRaisesRegex(AsrServiceError, "ffmpeg and ffprobe"),
        ):
            engine.load()


if __name__ == "__main__":
    unittest.main()
