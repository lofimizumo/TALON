"""Experiment stall watchdog for honest autonomous research runs.

Detects CPU-bound stalls (no log progress) and logs warnings; optional hard abort
when wall time exceeds a multiple of the pre-run estimate without heartbeat.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StallWatchdog:
    """Track L3/image progress and compare to budgeted runtime."""

    logger: logging.Logger
    estimate_sec: float
    label: str = "benchmark"
    heartbeat_timeout_sec: float = 600.0
    max_wall_factor: float = 3.0
    abort_on_stall: bool = False

    _last_heartbeat: float = field(default_factory=time.perf_counter, init=False)
    _started: float = field(default_factory=time.perf_counter, init=False)
    _images_done: int = 0
    _images_expected: int = 0

    def set_expected_images(self, n: int) -> None:
        self._images_expected = max(0, n)
        self.logger.info(
            "StallWatchdog[%s] expect ~%d L3 images est_total~%.0fs heartbeat=%.0fs",
            self.label,
            n,
            self.estimate_sec,
            self.heartbeat_timeout_sec,
        )

    def heartbeat(self, msg: str = "") -> None:
        self._last_heartbeat = time.perf_counter()
        if msg:
            self.logger.debug("heartbeat: %s", msg)

    def on_image_done(self, index: int, total: int, elapsed_sec: float) -> None:
        self._images_done += 1
        self.heartbeat()
        self.logger.info(
            "StallWatchdog progress %d/%d (%.2fs/img) %s",
            index,
            total,
            elapsed_sec,
            self.label,
        )
        self._check()

    def tick(self) -> None:
        """Call between seeds/paths."""
        self._check()

    def _check(self) -> None:
        now = time.perf_counter()
        idle = now - self._last_heartbeat
        wall = now - self._started
        if idle > self.heartbeat_timeout_sec:
            self.logger.warning(
                "STALL? %s: no heartbeat for %.0fs (wall=%.0fs est=%.0fs images=%d/%d)",
                self.label,
                idle,
                wall,
                self.estimate_sec,
                self._images_done,
                self._images_expected,
            )
            if self.abort_on_stall and idle > self.heartbeat_timeout_sec * 1.5:
                raise TimeoutError(
                    f"{self.label}: stalled >{self.heartbeat_timeout_sec:.0f}s without progress"
                )
        cap = self.estimate_sec * self.max_wall_factor
        if self.estimate_sec > 0 and wall > max(cap, 120.0):
            self.logger.warning(
                "STALL? %s: wall %.0fs > %.0fx estimate (%.0fs)",
                self.label,
                wall,
                self.max_wall_factor,
                self.estimate_sec,
            )
            if self.abort_on_stall and wall > cap * 2:
                raise TimeoutError(
                    f"{self.label}: exceeded {self.max_wall_factor}x estimated runtime"
                )

    @classmethod
    def from_env(
        cls,
        logger: logging.Logger,
        estimate_sec: float,
        *,
        label: str = "benchmark",
    ) -> StallWatchdog:
        timeout = float(os.environ.get("QFL_STALL_HEARTBEAT_SEC", "600"))
        factor = float(os.environ.get("QFL_STALL_WALL_FACTOR", "3"))
        abort = os.environ.get("QFL_ABORT_ON_STALL", "0") == "1"
        return cls(
            logger=logger,
            estimate_sec=estimate_sec,
            label=label,
            heartbeat_timeout_sec=timeout,
            max_wall_factor=factor,
            abort_on_stall=abort,
        )
