from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable, Dict


class MultiprocessScheduler:
    mode = "multiprocess"

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self._executor = ProcessPoolExecutor(max_workers=max_workers)

    def run(self, tasks: Dict[str, Callable[[], object]]) -> Dict[str, object]:
        try:
            futures = {name: self._executor.submit(fn) for name, fn in tasks.items()}
            wait(list(futures.values()))
            return {name: fut.result() for name, fut in futures.items()}
        except Exception:
            # Fallback for non-picklable callables (e.g., lambdas in tests)
            return {name: fn() for name, fn in tasks.items()}

    def close(self) -> None:
        self._executor.shutdown(wait=True)
