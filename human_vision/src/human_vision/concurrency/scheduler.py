from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from ..results import SchedulerDebug
from .threaded import ThreadedScheduler
from .multiprocess import MultiprocessScheduler


@dataclass
class SchedulerConfig:
    mode: str = "threaded"
    max_workers: int = 4


def build_scheduler(config: SchedulerConfig):
    if config.mode == "threaded":
        return ThreadedScheduler(max_workers=config.max_workers)
    if config.mode == "multiprocess":
        return MultiprocessScheduler(max_workers=config.max_workers)
    return ThreadedScheduler(max_workers=1)


def run_tasks(scheduler, tasks: Dict[str, Callable[[], object]]) -> tuple[Dict[str, object], SchedulerDebug]:
    debug = SchedulerDebug(mode=scheduler.mode, timings_ms={})
    results = scheduler.run(tasks)
    return results, debug
