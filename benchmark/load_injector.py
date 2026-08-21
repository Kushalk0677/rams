"""Minimal cross-core host-load generator for RAMS replay experiments.

This module intentionally imports only the standard library.  On Windows,
``multiprocessing`` uses spawn, so locating the worker here prevents every
worker from importing the controller, detector wrappers, and their optional
runtime dependencies before it can generate load.
"""

from __future__ import annotations

import multiprocessing
import os
import time


def _cpu_load_worker(stop_event, duty_value) -> None:
    """Consume one logical CPU at the shared duty fraction.

    The duty value is read once per 50 ms window.  The inner loop deliberately
    does not poll the multiprocessing event: that cross-process synchronization
    cost would materially reduce the intended CPU load.  Shutdown latency is
    therefore bounded by one window.
    """
    duty_window_s = 0.05
    accumulator = 0
    while not stop_event.is_set():
        with duty_value.get_lock():
            duty = max(0.0, min(1.0, duty_value.value))
        if duty <= 0.0:
            stop_event.wait(0.02)
            continue
        started = time.perf_counter()
        busy_until = started + duty_window_s * duty
        while time.perf_counter() < busy_until:
            accumulator = (accumulator * 1_103_515_245 + 12_345) & 0x7FFFFFFF
        remaining = duty_window_s - (time.perf_counter() - started)
        if remaining > 0:
            stop_event.wait(remaining)


class ProcessLoadInjector:
    """Generate a whole-host CPU target with isolated worker processes.

    One logical CPU is reserved by default for the controller.  Workers share
    a duty cycle scaled from the requested whole-host target.  Actual host
    telemetry remains the authoritative achieved-load record.
    """

    def __init__(self, intensity: float = 0.0, reserve_logical_cpus: int = 1):
        self.intensity = max(0.0, min(1.0, intensity))
        self.logical_cpus = max(1, os.cpu_count() or 1)
        self.reserve_logical_cpus = max(0, min(reserve_logical_cpus, self.logical_cpus - 1))
        self.worker_count = self.logical_cpus - self.reserve_logical_cpus if self.intensity > 0 else 0
        self._duty_scale = self.logical_cpus / max(1, self.worker_count)
        self._ctx = multiprocessing.get_context("spawn")
        self._stop = self._ctx.Event()
        self._duty = self._ctx.Value("d", 0.0)
        self._processes: list[multiprocessing.Process] = []
        self.set_intensity(self.intensity)

    def set_intensity(self, intensity: float) -> None:
        self.intensity = max(0.0, min(1.0, intensity))
        duty = min(1.0, self.intensity * self._duty_scale)
        with self._duty.get_lock():
            self._duty.value = duty

    def start(self) -> None:
        if self._processes:
            return
        for _ in range(self.worker_count):
            process = self._ctx.Process(target=_cpu_load_worker, args=(self._stop, self._duty))
            process.daemon = True
            process.start()
            self._processes.append(process)

    def stop(self) -> None:
        self._stop.set()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self._processes.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
