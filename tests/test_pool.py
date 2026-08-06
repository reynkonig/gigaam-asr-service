import contextlib
import os
import threading
import unittest
from typing import ClassVar
from unittest.mock import patch

from gigaam_asr_service.engine import AsrValidationError, TranscriptionResult
from gigaam_asr_service.pool import (
    AsrBusyError,
    AsrProcessPool,
    InProcessBackend,
    WorkerSnapshot,
    _worker_device,
)
from tests.helpers import make_settings


class Controller:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()


class FakeReplica:
    instances: ClassVar[list["FakeReplica"]] = []
    load_order: ClassVar[list[int]] = []
    controller: ClassVar[Controller | None] = None

    def __init__(self, settings, device, index, context):
        self.settings = settings
        self.device = device
        self.index = index
        self.context = context
        self.state = "cold"
        self.is_loaded = False
        self.last_error = None
        self.stopped = False
        self.error: Exception | None = None
        FakeReplica.instances.append(self)

    def start(self) -> None:
        return

    def load(self) -> None:
        FakeReplica.load_order.append(self.index)
        self.is_loaded = True
        self.state = "ready"

    def transcribe(self, _: str) -> TranscriptionResult:
        controller = FakeReplica.controller
        if controller is not None:
            controller.started.set()
            controller.release.wait(timeout=2)
        if self.error is not None:
            error = self.error
            self.error = None
            self.state = "ready"
            raise error
        self.state = "ready"
        return TranscriptionResult(f"worker-{self.index}", 1.0)

    def restart(self) -> None:
        self.stopped = False
        self.state = "cold"
        self.is_loaded = False

    def stop(self) -> None:
        self.stopped = True
        self.state = "stopped"
        self.is_loaded = False

    def snapshot(self) -> WorkerSnapshot:
        return WorkerSnapshot(
            index=self.index,
            device=self.device,
            pid=1000 + self.index,
            state=self.state,
            loaded=self.is_loaded,
            last_error=self.last_error,
        )


@contextlib.contextmanager
def fake_pool(settings, controller: Controller | None = None):
    FakeReplica.instances = []
    FakeReplica.load_order = []
    FakeReplica.controller = controller
    with patch("gigaam_asr_service.pool.ReplicaProcess", FakeReplica):
        pool = AsrProcessPool(settings)
        try:
            yield pool
        finally:
            if controller is not None:
                controller.release.set()
            pool.close()
            FakeReplica.controller = None


class DeviceIsolationTest(unittest.TestCase):
    def test_cuda_worker_selects_one_physical_device(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,7"}, clear=True):
            resolved = _worker_device("cuda:1")
            visible = os.environ["CUDA_VISIBLE_DEVICES"]
        self.assertEqual(resolved, "cuda:0")
        self.assertEqual(visible, "7")

    def test_out_of_range_inherited_device_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}, clear=True),
            self.assertRaisesRegex(Exception, "outside inherited"),
        ):
            _worker_device("cuda:1")


class ProcessPoolTest(unittest.TestCase):
    def test_requests_rotate_across_replicas(self) -> None:
        settings = make_settings(devices=("cuda:0", "cuda:1"))
        with fake_pool(settings) as pool:
            first = pool.transcribe("one.wav")
            second = pool.transcribe("two.wav")
        self.assertEqual(first.text, "worker-0")
        self.assertEqual(second.text, "worker-1")

    def test_validation_error_does_not_remove_healthy_worker(self) -> None:
        with fake_pool(make_settings()) as pool:
            pool.workers[0].error = AsrValidationError("bad audio")
            with self.assertRaisesRegex(AsrValidationError, "bad audio"):
                pool.transcribe("bad.wav")
            result = pool.transcribe("good.wav")
        self.assertEqual(result.text, "worker-0")

    def test_queue_capacity_rejects_excess_request_immediately(self) -> None:
        controller = Controller()
        settings = make_settings(max_pending_requests=0)
        with fake_pool(settings, controller) as pool:
            failures: list[BaseException] = []

            def run_first() -> None:
                try:
                    pool.transcribe("one.wav")
                except Exception as error:  # noqa: BLE001  # pragma: no cover
                    failures.append(error)

            thread = threading.Thread(target=run_first)
            thread.start()
            self.assertTrue(controller.started.wait(timeout=1))
            with self.assertRaisesRegex(AsrBusyError, "queue is full"):
                pool.transcribe("two.wav")
            controller.release.set()
            thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])

    def test_warmup_loads_replicas_sequentially(self) -> None:
        settings = make_settings(devices=("cuda:0", "cuda:1", "cuda:2"))
        with fake_pool(settings) as pool:
            pool.load()
            snapshot = pool.snapshot()
        self.assertEqual(FakeReplica.load_order, [0, 1, 2])
        self.assertEqual(snapshot["ready_workers"], 3)

    def test_busy_loaded_worker_remains_ready(self) -> None:
        with fake_pool(make_settings()) as pool:
            worker = pool.workers[0]
            worker.is_loaded = True
            worker.state = "busy"
            snapshot = pool.snapshot()
        self.assertEqual(snapshot["ready_workers"], 1)

    def test_close_stops_every_replica(self) -> None:
        settings = make_settings(devices=("cuda:0", "cuda:1"))
        with fake_pool(settings) as pool:
            workers = list(pool.workers)
        self.assertTrue(all(worker.stopped for worker in workers))


class InProcessBackendTest(unittest.TestCase):
    def test_multiple_devices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one device"):
            InProcessBackend(make_settings(devices=("cuda:0", "cuda:1")))


if __name__ == "__main__":
    unittest.main()
