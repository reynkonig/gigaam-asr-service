import contextlib
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gigaam_asr_service.config import MODEL_REVISION, MODEL_SOURCE
from gigaam_asr_service.engine import (
    AsrServiceError,
    AsrValidationError,
    GigaAmEngine,
)
from tests.helpers import make_settings


class FakeDevice:
    def __init__(self, value: str) -> None:
        self.value = value
        self.type = value.split(":", 1)[0]
        self.index = int(value.split(":", 1)[1]) if ":" in value else None

    def __str__(self) -> str:
        return self.value


class FakeCuda:
    def __init__(self, available: bool = True, count: int = 1) -> None:
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count


class FakeInnerModel:
    def __init__(self) -> None:
        self.forward = self.original_forward

    def original_forward(self, features, lengths):
        return features, lengths

    def preprocessor(self, features, lengths):
        return features, lengths

    def encoder(self, features, lengths):
        return features, lengths


class FakeModel:
    def __init__(self) -> None:
        self.model = FakeInnerModel()
        self.float_calls = 0
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.output: object = "распознанный текст"

    def float(self):
        self.float_calls += 1
        return self

    def to(self, device):
        self.to_calls.append(str(device))
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def transcribe(self, _: str) -> object:
        return self.output


def fake_runtime(
    model: FakeModel,
    *,
    cuda_available: bool = True,
    cuda_count: int = 1,
):
    torch_module = types.ModuleType("torch")
    torch_module.cuda = FakeCuda(cuda_available, cuda_count)
    torch_module.device = FakeDevice
    torch_module.inference_mode = contextlib.nullcontext

    calls: list[tuple[str, dict[str, object]]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(source: str, **kwargs: object) -> FakeModel:
            calls.append((source, kwargs))
            return model

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoModel = AutoModel
    return torch_module, transformers_module, calls


class GigaAmEngineTest(unittest.TestCase):
    def test_cuda_load_uses_pinned_model_and_forced_fp32_forward(self) -> None:
        model = FakeModel()
        torch_module, transformers_module, calls = fake_runtime(model)
        engine = GigaAmEngine(make_settings(devices=("cuda:0",)), "cuda:0")

        with (
            patch.dict(
                sys.modules,
                {"torch": torch_module, "transformers": transformers_module},
            ),
            patch("gigaam_asr_service.engine.shutil.which", return_value="/bin/tool"),
        ):
            engine.load()

        self.assertEqual(
            calls,
            [
                (
                    MODEL_SOURCE,
                    {
                        "revision": MODEL_REVISION,
                        "trust_remote_code": True,
                        "local_files_only": False,
                    },
                )
            ],
        )
        self.assertEqual(model.float_calls, 1)
        self.assertEqual(model.to_calls, ["cuda:0"])
        self.assertEqual(model.eval_calls, 1)
        self.assertIs(model.model.forward.__func__, GigaAmEngine._cuda_fp32_forward)
        self.assertEqual(engine.snapshot()["dtype"], "float32")
        self.assertEqual(engine.snapshot()["resolved_device"], "cuda:0")

    def test_cpu_load_keeps_official_forward(self) -> None:
        model = FakeModel()
        original = model.model.forward
        torch_module, transformers_module, _ = fake_runtime(
            model,
            cuda_available=False,
        )
        engine = GigaAmEngine(make_settings(), "cpu")
        with (
            patch.dict(
                sys.modules,
                {"torch": torch_module, "transformers": transformers_module},
            ),
            patch("gigaam_asr_service.engine.shutil.which", return_value="/bin/tool"),
        ):
            engine.load()
        self.assertEqual(model.model.forward, original)
        self.assertEqual(model.to_calls, ["cpu"])

    def test_unavailable_or_missing_cuda_device_is_rejected(self) -> None:
        model = FakeModel()
        for available, count, requested, message in (
            (False, 0, "cuda:0", "unavailable"),
            (True, 1, "cuda:2", "does not exist"),
        ):
            with self.subTest(requested=requested):
                torch_module, transformers_module, _ = fake_runtime(
                    model,
                    cuda_available=available,
                    cuda_count=count,
                )
                engine = GigaAmEngine(make_settings(devices=(requested,)), requested)
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "torch": torch_module,
                            "transformers": transformers_module,
                        },
                    ),
                    patch(
                        "gigaam_asr_service.engine.shutil.which",
                        return_value="/bin/tool",
                    ),
                    self.assertRaisesRegex(AsrServiceError, message),
                ):
                    engine.load()

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
                patch(
                    "gigaam_asr_service.engine.subprocess.run",
                    return_value=outcome,
                ),
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

    def test_transcription_is_trimmed_and_preserves_duration(self) -> None:
        model = FakeModel()
        model.output = "  готовый текст  "
        engine = GigaAmEngine(make_settings(), "cpu")
        engine._model = model
        engine._torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.object(engine, "_duration_seconds", return_value=2.5):
            result = engine.transcribe(Path("sample.wav"))
        self.assertEqual(result.text, "готовый текст")
        self.assertEqual(result.duration_seconds, 2.5)

    def test_invalid_model_output_is_service_failure(self) -> None:
        model = FakeModel()
        model.output = None
        engine = GigaAmEngine(make_settings(), "cpu")
        engine._model = model
        engine._torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        with (
            patch.object(engine, "_duration_seconds", return_value=1.0),
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
