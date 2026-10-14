from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import MODEL_REVISION, MODEL_SOURCE

MODEL_FILES = (
    "encoder.onnx",
    "decoder.onnx",
    "joint.onnx",
    "frontend.npz",
    "tokenizer.model",
    "manifest.json",
)


def model_directory() -> Path:
    home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "onnx" / MODEL_REVISION


def read_manifest(directory: Path) -> dict:
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(
            f"ONNX model is incomplete at {directory}; "
            "run the export-model command in README.md first"
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("source") != MODEL_SOURCE
        or manifest.get("revision") != MODEL_REVISION
        or manifest.get("dtype") != "float32"
    ):
        raise ValueError("ONNX model does not match the pinned FP32 model revision")
    return manifest


def extract_features(
    audio: np.ndarray, window: np.ndarray, mel_filters: np.ndarray, hop_length: int
) -> np.ndarray:
    frames = np.lib.stride_tricks.sliding_window_view(audio, window.size)[::hop_length]
    spectrum = np.fft.rfft(frames * window, axis=-1)
    power = np.abs(spectrum).astype(np.float32) ** 2
    mel = power @ mel_filters
    return np.log(np.clip(mel, 1e-9, 1e9)).T[np.newaxis].astype(np.float32)


class OnnxModel:
    def __init__(self, directory: Path, device: str) -> None:
        import onnxruntime as ort
        from sentencepiece import SentencePieceProcessor

        manifest = read_manifest(directory)
        with np.load(directory / "frontend.npz", allow_pickle=False) as frontend:
            self.window = frontend["window"]
            self.mel_filters = frontend["mel_filters"]
            self.hop_length = int(frontend["hop_length"])
        self.max_symbols = manifest["max_symbols_per_step"]
        self.tokenizer = SentencePieceProcessor(
            model_file=str(directory / "tokenizer.model")
        )
        self.blank_id = self.tokenizer.get_piece_size()

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers: list = ["CPUExecutionProvider"]
        if device != "cpu":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                ort.preload_dlls()
                index = int(device.split(":")[1]) if ":" in device else 0
                providers.insert(
                    0,
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": index,
                            "cudnn_conv_algo_search": "HEURISTIC",
                            "use_tf32": 0,
                        },
                    ),
                )
            elif device != "auto":
                raise ValueError(f"CUDA device {device!r} is unavailable")
        self.encoder = ort.InferenceSession(
            str(directory / "encoder.onnx"), options, providers=providers
        )
        uses_cuda = "CUDAExecutionProvider" in self.encoder.get_providers()
        if device.startswith("cuda") and not uses_cuda:
            raise ValueError(f"CUDA device {device!r} is unavailable or does not exist")
        self.resolved_device = (
            (device if device.startswith("cuda:") else "cuda:0") if uses_cuda else "cpu"
        )
        # The small prediction networks run on CPU to avoid a GPU round trip per token.
        self.decoder = ort.InferenceSession(
            str(directory / "decoder.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.joint = ort.InferenceSession(
            str(directory / "joint.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.state_shape = self.decoder.get_inputs()[1].shape

    def transcribe(self, audio: np.ndarray) -> str:
        features = extract_features(
            audio, self.window, self.mel_filters, self.hop_length
        )
        encoded, lengths = self.encoder.run(
            None,
            {
                "audio_signal": features,
                "length": np.array([features.shape[-1]], np.int64),
            },
        )
        return self.decode(encoded, int(lengths[0]))

    def decode(self, encoded: np.ndarray, length: int) -> str:
        tokens = []
        label = np.array([[self.blank_id]], dtype=np.int64)
        h = np.zeros(self.state_shape, dtype=np.float32)
        c = np.zeros_like(h)
        prediction = None
        for frame in range(length):
            for _ in range(self.max_symbols):
                if prediction is None:
                    prediction, next_h, next_c = self.decoder.run(
                        None, {"x": label, "h": h, "c": c}
                    )
                logits = self.joint.run(
                    None,
                    {
                        "enc": encoded[:, :, frame : frame + 1],
                        "dec": prediction.transpose(0, 2, 1),
                    },
                )[0]
                token = int(logits.argmax())
                if token == self.blank_id:
                    break
                tokens.append(token)
                label[0, 0] = token
                h, c = next_h, next_c
                prediction = None
        return self.tokenizer.decode(tokens)
