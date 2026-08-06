from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from .config import ServerSettings
from .engine import (
    AsrServiceError,
    AsrValidationError,
    GigaAmEngine,
    TranscriptionResult,
)

LOGGER = logging.getLogger(__name__)
CUDA_INDEX_PATTERN = re.compile(r"cuda(?::(\d+))?\Z")


class AsrBusyError(RuntimeError):
    pass


class AsrBackend(Protocol):
    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def transcribe(self, audio_path: str) -> TranscriptionResult: ...

    def snapshot(self) -> dict[str, object]: ...

    def close(self) -> None: ...


def _worker_device(requested_device: str) -> str:
    match = CUDA_INDEX_PATTERN.fullmatch(requested_device)
    if match is None:
        return requested_device

    requested_index = int(match.group(1) or "0")
    inherited_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited_devices is None:
        selected_device = str(requested_index)
    else:
        visible_devices = tuple(
            item.strip()
            for item in inherited_devices.split(",")
            if item.strip() and item.strip() != "-1"
        )
        if requested_index >= len(visible_devices):
            raise AsrServiceError(
                f"CUDA device {requested_device!r} is outside inherited "
                f"CUDA_VISIBLE_DEVICES={inherited_devices!r}"
            )
        selected_device = visible_devices[requested_index]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
    return "cuda:0"


def _worker_main(
    settings: ServerSettings,
    requested_device: str,
    command_queue: object,
    response_queue: object,
) -> None:
    runtime_device = _worker_device(requested_device)
    engine = GigaAmEngine(settings, runtime_device)
    while True:
        command = command_queue.get()  # type: ignore[attr-defined]
        if not isinstance(command, tuple) or not command:
            continue
        operation = command[0]
        if operation == "shutdown":
            return
        try:
            if operation == "load":
                engine.load()
                result = None
            elif operation == "transcribe":
                result = engine.transcribe(command[1])
            else:
                raise RuntimeError(f"unsupported worker operation: {operation!r}")
            response_queue.put(("ok", result, engine.snapshot()))  # type: ignore[attr-defined]
        except AsrValidationError as error:
            response_queue.put(  # type: ignore[attr-defined]
                ("validation_error", str(error), engine.snapshot())
            )
        except AsrServiceError as error:
            response_queue.put(  # type: ignore[attr-defined]
                ("error", str(error), engine.snapshot())
            )
        except Exception as error:
            LOGGER.exception("Unhandled GigaAM worker failure")
            response_queue.put(  # type: ignore[attr-defined]
                (
                    "error",
                    f"GigaAM worker failed with {type(error).__name__}",
                    engine.snapshot(),
                )
            )


@dataclass
class WorkerSnapshot:
    index: int
    device: str
    pid: int | None
    state: str
    loaded: bool
    last_error: str | None


class ReplicaProcess:
    def __init__(
        self,
        settings: ServerSettings,
        device: str,
        index: int,
        context: multiprocessing.context.BaseContext,
    ) -> None:
        self.settings = settings
        self.device = device
        self.index = index
        self.context = context
        self._command_queue: object | None = None
        self._response_queue: object | None = None
        self._process: multiprocessing.Process | None = None
        self._lock = threading.Lock()
        self.state = "cold"
        self.is_loaded = False
        self.last_error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return
            self._command_queue = self.context.Queue(maxsize=1)
            self._response_queue = self.context.Queue(maxsize=1)
            self._process = self.context.Process(
                target=_worker_main,
                args=(
                    self.settings,
                    self.device,
                    self._command_queue,
                    self._response_queue,
                ),
                name=f"gigaam-{self.index}-{self.device.replace(':', '-')}",
                daemon=True,
            )
            self._process.start()
            self.state = "cold"
            self.is_loaded = False
            self.last_error = None

    def _terminate_after_timeout(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

    def execute(
        self,
        operation: str,
        payload: object | None = None,
    ) -> TranscriptionResult | None:
        self.start()
        assert self._command_queue is not None
        assert self._response_queue is not None
        assert self._process is not None

        self.state = "loading" if operation == "load" or not self.is_loaded else "busy"
        command = (operation,) if payload is None else (operation, payload)
        self._command_queue.put(command)  # type: ignore[attr-defined]
        deadline = time.monotonic() + self.settings.inference_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.state = "failed"
                self.last_error = "WorkerTimeout"
                self._terminate_after_timeout()
                raise AsrServiceError(
                    f"GigaAM worker {self.device} exceeded the inference timeout"
                )
            try:
                response = self._response_queue.get(  # type: ignore[attr-defined]
                    timeout=min(0.5, remaining)
                )
                break
            except queue.Empty:
                if not self._process.is_alive():
                    self.state = "failed"
                    self.last_error = "WorkerExited"
                    raise AsrServiceError(
                        f"GigaAM worker {self.device} exited unexpectedly"
                    )

        outcome, value, snapshot = response
        if isinstance(snapshot, dict):
            self.is_loaded = bool(snapshot.get("loaded"))
            raw_error = snapshot.get("last_error")
            self.last_error = str(raw_error) if raw_error else None
        if outcome == "validation_error":
            self.state = "ready" if self.is_loaded else "cold"
            self.last_error = None
            raise AsrValidationError(str(value))
        if outcome == "error":
            self.state = "failed"
            self.last_error = self.last_error or "AsrServiceError"
            raise AsrServiceError(str(value))
        self.state = "ready"
        self.last_error = None
        return value if isinstance(value, TranscriptionResult) else None

    def load(self) -> None:
        self.execute("load")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        result = self.execute("transcribe", audio_path)
        if result is None:
            raise AsrServiceError(
                f"GigaAM worker {self.device} returned no transcription"
            )
        return result

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        with self._lock:
            process = self._process
            command_queue = self._command_queue
            if process is None:
                return
            if process.is_alive() and command_queue is not None:
                try:
                    command_queue.put(("shutdown",), timeout=1)  # type: ignore[attr-defined]
                except queue.Full:
                    pass
            process.join(timeout=self.settings.shutdown_timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            for channel in (self._command_queue, self._response_queue):
                close = getattr(channel, "close", None)
                if close is not None:
                    close()
            self._process = None
            self._command_queue = None
            self._response_queue = None
            self.state = "stopped"
            self.is_loaded = False

    def snapshot(self) -> WorkerSnapshot:
        process = self._process
        if (
            process is not None
            and not process.is_alive()
            and self.state not in {"cold", "stopped"}
        ):
            self.state = "failed"
            self.last_error = self.last_error or "WorkerExited"
        return WorkerSnapshot(
            index=self.index,
            device=self.device,
            pid=process.pid if process is not None else None,
            state=self.state,
            loaded=self.is_loaded,
            last_error=self.last_error,
        )


class AsrProcessPool:
    def __init__(
        self,
        settings: ServerSettings,
        *,
        context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        self.settings = settings
        self.context = context or multiprocessing.get_context("spawn")
        self.workers = [
            ReplicaProcess(settings, device, index, self.context)
            for index, device in enumerate(settings.devices)
        ]
        self._available: queue.Queue[ReplicaProcess] = queue.Queue(
            maxsize=len(self.workers)
        )
        for worker in self.workers:
            self._available.put_nowait(worker)
        self._capacity = threading.BoundedSemaphore(
            len(self.workers) + settings.max_pending_requests
        )
        self._state_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._model_load_lock = threading.Lock()
        self._started = False
        self._closing = False
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._restart_timers: set[threading.Timer] = set()

    @property
    def loaded(self) -> bool:
        return bool(self.workers) and all(worker.is_loaded for worker in self.workers)

    def start(self) -> None:
        with self._state_lock:
            if self._closing:
                raise AsrServiceError("GigaAM pool is shutting down")
            if self._started:
                return
            for worker in self.workers:
                worker.start()
            self._started = True

    def _restart_and_return(self, worker: ReplicaProcess) -> None:
        timer = threading.current_thread()
        try:
            with self._state_lock:
                closing = self._closing
            if closing:
                return
            worker.restart()
            with self._state_lock:
                closing = self._closing
                if not closing:
                    self._available.put_nowait(worker)
            if closing:
                worker.stop()
        except Exception:
            LOGGER.exception("Failed to restart GigaAM worker %s", worker.device)
        finally:
            if isinstance(timer, threading.Timer):
                with self._state_lock:
                    self._restart_timers.discard(timer)

    def _release_worker(self, worker: ReplicaProcess) -> None:
        if self._closing:
            worker.stop()
            return
        if worker.state != "failed":
            self._available.put(worker)
            return
        timer = threading.Timer(
            self.settings.worker_restart_cooldown_seconds,
            self._restart_and_return,
            args=(worker,),
        )
        timer.daemon = True
        with self._state_lock:
            self._restart_timers.add(timer)
        timer.start()

    def _acquire_worker(self) -> ReplicaProcess:
        if not self._capacity.acquire(blocking=False):
            raise AsrBusyError("transcription request queue is full")
        try:
            self.start()
            return self._available.get(timeout=self.settings.queue_timeout_seconds)
        except queue.Empty as error:
            self._capacity.release()
            raise AsrBusyError(
                "timed out waiting for an available GigaAM worker"
            ) from error
        except BaseException:
            self._capacity.release()
            raise

    def _finish_request(self, worker: ReplicaProcess, *, failed: bool) -> None:
        with self._state_lock:
            self._active -= 1
            if failed:
                self._failed += 1
            else:
                self._completed += 1
        self._release_worker(worker)
        self._capacity.release()

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        worker = self._acquire_worker()
        with self._state_lock:
            self._active += 1
        try:
            self._ensure_worker_loaded(worker)
            result = worker.transcribe(audio_path)
        except BaseException:
            self._finish_request(worker, failed=True)
            raise
        self._finish_request(worker, failed=False)
        return result

    def _ensure_worker_loaded(self, worker: ReplicaProcess) -> None:
        if worker.is_loaded:
            return
        if not self._model_load_lock.acquire(
            timeout=self.settings.inference_timeout_seconds
        ):
            raise AsrServiceError("timed out waiting to load a GigaAM replica")
        try:
            if not worker.is_loaded:
                worker.load()
        finally:
            self._model_load_lock.release()

    def load(self) -> None:
        if not self._warmup_lock.acquire(timeout=self.settings.queue_timeout_seconds):
            raise AsrServiceError("timed out waiting for another GigaAM warmup")
        try:
            self.start()
            failures: list[Exception] = []
            reserved: list[ReplicaProcess] = []
            deadline = time.monotonic() + self.settings.queue_timeout_seconds
            try:
                for _ in self.workers:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    reserved.append(self._available.get(timeout=remaining))
            except queue.Empty as error:
                for worker in reserved:
                    self._release_worker(worker)
                raise AsrServiceError(
                    "timed out reserving GigaAM workers for warmup"
                ) from error

            for worker in reserved:
                try:
                    self._ensure_worker_loaded(worker)
                except Exception as error:  # noqa: BLE001
                    failures.append(error)
                finally:
                    self._release_worker(worker)
            if len(failures) == len(self.workers):
                raise AsrServiceError("all GigaAM workers failed to preload")
        finally:
            self._warmup_lock.release()

    def snapshot(self) -> dict[str, object]:
        worker_rows = [worker.snapshot().__dict__ for worker in self.workers]
        ready = sum(
            bool(row["loaded"]) and row["state"] in {"ready", "busy"}
            for row in worker_rows
        )
        failed = sum(row["state"] == "failed" for row in worker_rows)
        with self._state_lock:
            active = self._active
            completed = self._completed
            request_failures = self._failed
        return {
            "mode": "replicated",
            "workers": worker_rows,
            "worker_count": len(worker_rows),
            "ready_workers": ready,
            "failed_workers": failed,
            "active_requests": active,
            "available_workers": self._available.qsize(),
            "completed_requests": completed,
            "failed_requests": request_failures,
        }

    def close(self) -> None:
        with self._state_lock:
            if self._closing:
                return
            self._closing = True
            self._started = False
            timers = list(self._restart_timers)
            self._restart_timers.clear()
        for timer in timers:
            timer.cancel()
        for worker in self.workers:
            worker.stop()


class InProcessBackend:
    def __init__(self, settings: ServerSettings) -> None:
        if len(settings.devices) != 1:
            raise ValueError("in-process backend supports exactly one device")
        self.engine = GigaAmEngine(settings, settings.devices[0])

    @property
    def loaded(self) -> bool:
        return self.engine.loaded

    def load(self) -> None:
        self.engine.load()

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        return self.engine.transcribe(audio_path)

    def snapshot(self) -> dict[str, object]:
        engine = self.engine.snapshot()
        ready = int(self.loaded)
        return {
            "mode": "in-process",
            "workers": [engine],
            "worker_count": 1,
            "ready_workers": ready,
            "failed_workers": int(engine["state"] == "failed"),
            "active_requests": 0,
            "available_workers": 1,
            "completed_requests": 0,
            "failed_requests": 0,
        }

    def close(self) -> None:
        return
