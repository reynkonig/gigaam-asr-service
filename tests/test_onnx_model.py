import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from gigaam_asr_service.config import MODEL_REVISION, MODEL_SOURCE
from gigaam_asr_service.export_onnx import export_model
from gigaam_asr_service.onnx_model import (
    MODEL_FILES,
    OnnxModel,
    extract_features,
    model_directory,
    read_manifest,
)


class OnnxModelTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for name in MODEL_FILES:
            (self.directory / name).touch()
        self.manifest = {
            "source": MODEL_SOURCE,
            "revision": MODEL_REVISION,
            "dtype": "float32",
            "max_symbols_per_step": 10,
        }
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))
        np.savez(
            self.directory / "frontend.npz",
            window=np.ones(4, np.float32),
            mel_filters=np.eye(3, dtype=np.float32),
            hop_length=2,
        )

    def test_features_preserve_power_log_scaling_and_frame_count(self) -> None:
        features = extract_features(
            np.ones(8, np.float32),
            np.ones(4, np.float32),
            np.eye(3, dtype=np.float32),
            2,
        )
        expected = np.tile(
            np.log(np.array([16, 1e-9, 1e-9], np.float32))[:, None], (1, 3)
        )
        np.testing.assert_allclose(features[0], expected)
        self.assertEqual(features.dtype, np.float32)
        silence = extract_features(
            np.zeros(4, np.float32),
            np.ones(4, np.float32),
            np.eye(3, dtype=np.float32),
            2,
        )
        self.assertTrue(np.isfinite(silence).all())

    def test_model_directory_uses_existing_hugging_face_cache(self) -> None:
        with patch.dict(os.environ, {"HF_HOME": "/cache/huggingface"}):
            self.assertEqual(
                model_directory(), Path("/cache/huggingface/onnx") / MODEL_REVISION
            )

    def test_manifest_rejects_other_model_revisions_and_partial_exports(self) -> None:
        self.manifest["revision"] = "0" * 40
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "pinned FP32"):
            read_manifest(self.directory)
        (self.directory / "joint.onnx").unlink()
        with self.assertRaisesRegex(ValueError, "export-model"):
            read_manifest(self.directory)

    def test_repeated_export_reuses_completed_artifacts_without_torch(self) -> None:
        with patch(
            "gigaam_asr_service.export_onnx.model_directory",
            return_value=self.directory,
        ):
            self.assertEqual(export_model(local_files_only=True), self.directory)

    def test_failed_export_never_publishes_partial_model(self) -> None:
        destination = self.directory / "new-model"
        module = Mock()
        module.input_example.return_value = ()
        module.input_names.return_value = []
        module.output_names.return_value = []
        model = SimpleNamespace(
            encoder=module,
            head=SimpleNamespace(decoder=module, joint=module),
            preprocessor=SimpleNamespace(
                featurizer=[
                    SimpleNamespace(
                        spectrogram=SimpleNamespace(
                            center=False, win_length=320, n_fft=320
                        )
                    )
                ]
            ),
        )
        pretrained = Mock()
        pretrained.float.return_value.cpu.return_value.eval.return_value.model = model
        torch = SimpleNamespace(
            set_num_threads=Mock(),
            inference_mode=contextlib.nullcontext,
            onnx=SimpleNamespace(
                export=Mock(side_effect=RuntimeError("export failed"))
            ),
        )
        transformers = SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=Mock(return_value=pretrained))
        )
        with (
            patch(
                "gigaam_asr_service.export_onnx.model_directory",
                return_value=destination,
            ),
            patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
            self.assertRaisesRegex(RuntimeError, "export failed"),
        ):
            export_model(local_files_only=True)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.directory.glob(".export-*")), [])

    def load_model(self, device: str, available: list[str], selected: list[str]):
        session = SimpleNamespace(
            get_providers=lambda: selected,
            get_inputs=lambda: [None, SimpleNamespace(shape=[1, 1, 320])],
        )
        with (
            patch("onnxruntime.get_available_providers", return_value=available),
            patch("onnxruntime.preload_dlls"),
            patch("onnxruntime.InferenceSession", return_value=session),
            patch("sentencepiece.SentencePieceProcessor") as tokenizer,
        ):
            tokenizer.return_value.get_piece_size.return_value = 1024
            return OnnxModel(self.directory, device)

    def test_requested_cuda_cannot_silently_fall_back_to_cpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.load_model(
                "cuda:0", ["CPUExecutionProvider"], ["CPUExecutionProvider"]
            )
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.load_model(
                "cuda:2", ["CUDAExecutionProvider"], ["CPUExecutionProvider"]
            )

    def test_auto_can_use_cpu_when_cuda_provider_cannot_load(self) -> None:
        model = self.load_model(
            "auto", ["CUDAExecutionProvider"], ["CPUExecutionProvider"]
        )
        self.assertEqual(model.resolved_device, "cpu")

    def decoder_model(self, labels: list[int], max_symbols: int = 10):
        model = OnnxModel.__new__(OnnxModel)
        model.state_shape = [1, 1, 1]
        model.blank_id = 3
        model.max_symbols = max_symbols
        model.tokenizer = SimpleNamespace(decode=lambda tokens: tokens)
        calls = []

        def predict(_, inputs):
            calls.append((int(inputs["x"][0, 0]), float(inputs["h"].item())))
            return np.zeros((1, 1, 1)), inputs["h"] + 1, inputs["c"] + 1

        def joint(_, inputs):
            logits = np.zeros(4)
            logits[labels.pop(0)] = 1
            return [logits]

        model.decoder = SimpleNamespace(run=predict)
        model.joint = SimpleNamespace(run=joint)
        return model, calls

    def test_rnnt_blank_does_not_advance_prediction_state(self) -> None:
        model, calls = self.decoder_model([1, 3, 2, 3])
        self.assertEqual(model.decode(np.zeros((1, 1, 2)), 2), [1, 2])
        self.assertEqual(calls, [(3, 0), (1, 1), (2, 2)])

    def test_rnnt_caps_symbols_per_frame_and_respects_encoder_length(self) -> None:
        model, _ = self.decoder_model([1, 1, 2, 3], max_symbols=2)
        self.assertEqual(model.decode(np.zeros((1, 1, 4)), 2), [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
