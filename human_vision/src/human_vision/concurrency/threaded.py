from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Dict


class ThreadedScheduler:
    mode = "threaded"

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def run(self, tasks: Dict[str, Callable[[], object]]) -> Dict[str, object]:
        futures = {name: self._executor.submit(fn) for name, fn in tasks.items()}
        wait(list(futures.values()))
        return {name: fut.result() for name, fut in futures.items()}

    def close(self) -> None:
        self._executor.shutdown(wait=True)
