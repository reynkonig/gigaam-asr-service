import os
import unittest
from unittest.mock import patch

from gigaam_asr_service.cli import build_parser, settings_from_args


class CommandLineConfigurationTest(unittest.TestCase):
    def parse(self, *arguments: str):
        return build_parser().parse_args(list(arguments))

    def test_device_alias_and_multiple_replicas_share_one_setting(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            single = settings_from_args(self.parse("--device", "cuda"))
            multiple = settings_from_args(self.parse("--devices", "cuda:0,cuda:2"))
        self.assertEqual(single.devices, ("cuda:0",))
        self.assertEqual(multiple.devices, ("cuda:0", "cuda:2"))

    def test_cli_overrides_environment_before_validation(self) -> None:
        environment = {
            "ASR_DEVICE": "cpu",
            "ASR_MAX_AUDIO_SECONDS": "20",
            "ASR_MAX_UPLOAD_BYTES": "100",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = settings_from_args(
                self.parse(
                    "--device",
                    "cuda:1",
                    "--max-audio-seconds",
                    "22",
                    "--max-upload-bytes",
                    "200",
                )
            )
        self.assertEqual(settings.devices, ("cuda:1",))
        self.assertEqual(settings.max_audio_seconds, 22)
        self.assertEqual(settings.max_upload_bytes, 200)

    def test_cli_does_not_expose_model_selection(self) -> None:
        parser = build_parser()
        options = {
            option for action in parser._actions for option in action.option_strings
        }
        self.assertNotIn("--model", options)
        self.assertNotIn("--revision", options)

    def test_cli_validation_runs_after_overrides(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "max_audio_seconds"),
        ):
            settings_from_args(self.parse("--max-audio-seconds", "25"))


if __name__ == "__main__":
    unittest.main()
