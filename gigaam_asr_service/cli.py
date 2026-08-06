from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from .api import create_app
from .config import ServerSettings
from .pool import InProcessBackend


def _device_list(value: str) -> tuple[str, ...]:
    devices = tuple(
        "cuda:0" if item.strip() == "cuda" else item.strip()
        for item in value.split(",")
        if item.strip()
    )
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    return devices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve GigaAM v3 E2E RNN-T through an OpenAI-compatible API",
    )
    parser.add_argument("--host", help="listen address (environment default)")
    parser.add_argument("--port", type=int, help="listen port")
    parser.add_argument(
        "--devices",
        "--device",
        dest="devices",
        type=_device_list,
        metavar="DEVICE[,DEVICE...]",
        help="model replica devices, for example cuda:0,cuda:1",
    )
    parser.add_argument("--max-upload-bytes", type=int)
    parser.add_argument("--max-audio-seconds", type=float)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="debug mode for one device; disables process isolation",
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    return parser


def settings_from_args(args: argparse.Namespace) -> ServerSettings:
    updates: dict[str, object] = {}
    for field in (
        "host",
        "port",
        "devices",
        "max_upload_bytes",
        "max_audio_seconds",
        "local_files_only",
        "preload",
    ):
        value = getattr(args, field)
        if value is not None:
            updates[field] = value
    return ServerSettings.from_env(**updates)


def main(argv: Sequence[str] | None = None) -> None:
    import uvicorn

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        backend = InProcessBackend(settings) if args.in_process else None
    except ValueError as error:
        parser.error(str(error))

    uvicorn.run(
        create_app(settings, backend=backend),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
