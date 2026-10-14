from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .config import MODEL_REVISION, MODEL_SOURCE
from .onnx_model import (
    MODEL_FILES,
    OnnxModel,
    extract_features,
    model_directory,
    read_manifest,
)


def export_model(*, local_files_only: bool = False) -> Path:
    directory = model_directory()
    if directory.exists():
        read_manifest(directory)
        print(f"ONNX model already exported: {directory}")
        return directory

    import numpy as np
    import torch
    from transformers import AutoModel

    torch.set_num_threads(1)
    model = (
        AutoModel.from_pretrained(
            MODEL_SOURCE,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        .float()
        .cpu()
        .eval()
        .model
    )
    mel_transform = model.preprocessor.featurizer[0]
    spectrogram = mel_transform.spectrogram
    if spectrogram.center or spectrogram.win_length != spectrogram.n_fft:
        raise ValueError("Unexpected preprocessor in the pinned model")

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".export-", dir=directory.parent
    ) as temporary:
        output = Path(temporary)
        modules = (
            ("encoder", model.encoder),
            ("decoder", model.head.decoder),
            ("joint", model.head.joint),
        )
        with torch.inference_mode():
            for name, module in modules:
                inputs = module.input_example()
                if name == "encoder":
                    module(*inputs)  # Initialize positional embeddings before tracing.
                torch.onnx.export(
                    module,
                    inputs,
                    str(output / f"{name}.onnx"),
                    input_names=module.input_names(),
                    output_names=(
                        ["dec", "next_h", "next_c"]
                        if name == "decoder"
                        else module.output_names()
                    ),
                    dynamic_axes=(
                        {
                            "audio_signal": {2: "seq_len"},
                            "encoded": {2: "encoded_seq_len"},
                        }
                        if name == "encoder"
                        else None
                    ),
                    opset_version=17,
                    dynamo=False,
                )

        window = spectrogram.window.cpu().numpy()
        mel_filters = mel_transform.mel_scale.fb.cpu().numpy()
        np.savez(
            output / "frontend.npz",
            window=window,
            mel_filters=mel_filters,
            hop_length=spectrogram.hop_length,
        )
        shutil.copyfile(model.cfg.decoding.model_path, output / "tokenizer.model")
        manifest = {
            "source": MODEL_SOURCE,
            "revision": MODEL_REVISION,
            "dtype": "float32",
            "max_symbols_per_step": model.decoding.max_symbols,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        exported = OnnxModel(output, "cpu")
        with torch.inference_mode():
            for name, module in modules[1:]:
                inputs = module.input_example()
                expected = module(*inputs)
                actual = getattr(exported, name).run(
                    None,
                    dict(
                        zip(
                            module.input_names(),
                            (x.numpy() for x in inputs),
                            strict=True,
                        )
                    ),
                )
                if isinstance(expected, torch.Tensor):
                    expected = (expected,)
                for value, reference in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(
                        value, reference.numpy(), atol=1e-5, rtol=1e-4
                    )
            for size in (320, 16000, 37123):
                audio = (
                    np.random.default_rng(size).normal(0, 0.1, size).astype(np.float32)
                )
                expected_features, lengths = model.preprocessor(
                    torch.from_numpy(audio).unsqueeze(0), torch.tensor([size])
                )
                features = extract_features(
                    audio, window, mel_filters, spectrogram.hop_length
                )
                np.testing.assert_allclose(
                    features, expected_features.numpy(), atol=2e-5, rtol=2e-5
                )
                expected, expected_lengths = model.encoder(expected_features, lengths)
                actual, actual_lengths = exported.encoder.run(
                    None, {"audio_signal": features, "length": lengths.numpy()}
                )
                np.testing.assert_array_equal(actual_lengths, expected_lengths.numpy())
                np.testing.assert_allclose(
                    actual, expected.numpy(), atol=2e-3, rtol=2e-3
                )
        # The non-root server can have a different UID from the exporter.
        for name in MODEL_FILES:
            (output / name).chmod(0o644)
        output.chmod(0o755)
        output.rename(directory)
    print(f"Exported and checked FP32 ONNX model: {directory}")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the pinned GigaAM model to ONNX"
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    export_model(local_files_only=args.local_files_only)


if __name__ == "__main__":
    main()
