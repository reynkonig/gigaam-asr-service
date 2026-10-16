import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gigaam_asr_service.engine import AsrServiceError, AsrValidationError, GigaAmEngine
from tests.helpers import make_settings, wav_bytes


class FakeModel:
    resolved_device = "cpu"
    output: object = "распознанный текст"

    def transcribe(self, audio: np.ndarray) -> object:
        return self.output


class GigaAmEngineTest(unittest.TestCase):
    def audio_path(self, content: bytes) -> Path:
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(directory) / "upload.audio"
        path.write_bytes(content)
        return path

    def test_load_uses_onnx_artifacts_and_requested_device_once(self) -> None:
        model = FakeModel()
        model.resolved_device = "cuda:0"
        engine = GigaAmEngine(make_settings(devices=("cuda:0",)), "cuda:0")
        with (
            patch(
                "gigaam_asr_service.engine.model_directory", return_value=Path("cache")
            ),
            patch("gigaam_asr_service.engine.OnnxModel", return_value=model) as factory,
        ):
            engine.load()
            engine.load()
        factory.assert_called_once_with(Path("cache"), "cuda:0")
        self.assertEqual(engine.snapshot()["dtype"], "float32")
        self.assertEqual(engine.snapshot()["resolved_device"], "cuda:0")

    def test_load_failure_is_reported_and_can_be_retried(self) -> None:
        engine = GigaAmEngine(make_settings(), "cpu")
        with (
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
            patch("gigaam_asr_service.engine.OnnxModel", return_value=FakeModel()),
        ):
            engine.load()
        self.assertEqual(engine.snapshot()["state"], "ready")
        self.assertIsNone(engine.snapshot()["last_error"])

    def test_prepared_samples_are_read_without_external_commands(self) -> None:
        model = FakeModel()
        engine = GigaAmEngine(make_settings(), "cpu")
        pcm = np.tile(np.array([-32768, -1, 0, 1, 32767], dtype="<i2"), 64)
        path = self.audio_path(wav_bytes(pcm.tobytes()))
        with (
            patch("subprocess.run", side_effect=AssertionError("external command")),
            patch("gigaam_asr_service.engine.OnnxModel", return_value=model),
            patch.object(
                model, "transcribe", return_value="  готовый текст  "
            ) as infer,
        ):
            result = engine.transcribe(path)
        actual = infer.call_args.args[0]
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, pcm.astype(np.float32) / 32768.0)
        self.assertEqual(result.text, "готовый текст")
        self.assertEqual(result.duration_seconds, 0.02)

    def test_exact_duration_limit_is_accepted(self) -> None:
        engine = GigaAmEngine(make_settings(max_audio_seconds=1), "cpu")
        pcm = np.full(16000, -32768, dtype="<i2").tobytes()
        audio = engine._load_audio(str(self.audio_path(wav_bytes(pcm))))
        np.testing.assert_array_equal(audio, np.full(16000, -1, dtype=np.float32))

    def test_duration_limits_are_checked_before_loading_model(self) -> None:
        engine = GigaAmEngine(make_settings(max_audio_seconds=1), "cpu")
        for frames, message in (
            (0, "20 ms"),
            (319, "20 ms"),
            (16001, "duration exceeds"),
        ):
            path = self.audio_path(wav_bytes(b"\0\0" * frames))
            with (
                self.subTest(frames=frames),
                patch("gigaam_asr_service.engine.OnnxModel") as model,
                self.assertRaisesRegex(AsrValidationError, message),
            ):
                engine.transcribe(path)
            model.assert_not_called()

    def test_files_needing_conversion_are_rejected(self) -> None:
        engine = GigaAmEngine(make_settings(), "cpu")
        for parameters in (
            {"channels": 2},
            {"sample_rate": 8000},
            {"sample_rate": 44100},
            {"sample_width": 1},
            {"sample_width": 3},
            {"sample_width": 4},
        ):
            path = self.audio_path(wav_bytes(b"\0" * 960, **parameters))
            with (
                self.subTest(parameters=parameters),
                self.assertRaisesRegex(AsrValidationError, "mono 16 kHz 16-bit PCM"),
            ):
                engine._load_audio(str(path))

    def test_corrupt_and_truncated_files_are_rejected(self) -> None:
        engine = GigaAmEngine(make_settings(), "cpu")
        valid = wav_bytes(b"\0\0" * 320)
        floating_point = bytearray(valid)
        struct.pack_into("<H", floating_point, valid.index(b"fmt ") + 8, 3)
        oversized_chunk = bytearray(valid)
        struct.pack_into("<I", oversized_chunk, valid.index(b"fmt ") + 4, 1000)
        for payload in (
            b"",
            b"not WAV",
            valid[:20],
            valid[:-1],
            floating_point,
            oversized_chunk,
        ):
            with (
                self.subTest(size=len(payload)),
                self.assertRaises(AsrValidationError),
            ):
                engine._load_audio(str(self.audio_path(payload)))

    def test_incomplete_sample_is_rejected(self) -> None:
        engine = GigaAmEngine(make_settings(), "cpu")
        malformed = bytearray(wav_bytes(b"\0\0" * 320))
        malformed.append(0)
        struct.pack_into("<I", malformed, 4, len(malformed) - 8)
        struct.pack_into("<I", malformed, malformed.index(b"data") + 4, 641)
        with self.assertRaisesRegex(AsrValidationError, "declared frame count"):
            engine._load_audio(str(self.audio_path(malformed)))

    def test_missing_file_is_validation_error(self) -> None:
        engine = GigaAmEngine(make_settings(), "cpu")
        with self.assertRaises(AsrValidationError):
            engine._load_audio("/no-such-audio.wav")

    def test_transcription_duration_comes_from_sample_count(self) -> None:
        model = FakeModel()
        model.output = "  готовый текст  "
        engine = GigaAmEngine(make_settings(), "cpu")
        engine._model = model
        with (
            patch.object(
                engine, "_load_audio", return_value=np.zeros(40000, np.float32)
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
            patch.object(
                engine, "_load_audio", return_value=np.zeros(16000, np.float32)
            ),
            self.assertRaisesRegex(AsrServiceError, "invalid transcription"),
        ):
            engine.transcribe("sample.wav")


if __name__ == "__main__":
    unittest.main()
