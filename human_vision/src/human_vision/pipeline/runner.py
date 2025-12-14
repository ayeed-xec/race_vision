from __future__ import annotations

from typing import Dict, List

from ..config import VisionConfig
from ..concurrency.scheduler import SchedulerConfig, build_scheduler, run_tasks
from ..core.types import Capability
from ..results import VisionDebug
from .stages import Stage


class PipelineRunner:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.scheduler = build_scheduler(
            SchedulerConfig(mode=config.runtime.mode, max_workers=config.runtime.max_workers)
        )

    def run(self, stages: List[Stage]) -> tuple[Dict[Capability, object], VisionDebug]:
        tasks = {stage.name: stage for stage in stages}
        results, scheduler_debug = run_tasks(self.scheduler, tasks)
        outputs: Dict[Capability, object] = {}
        for stage in stages:
            outputs[stage.capability] = results.get(stage.name)
        debug = VisionDebug(scheduler=scheduler_debug)
        return outputs, debug

    def close(self) -> None:
        self.scheduler.close()
