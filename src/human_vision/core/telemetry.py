from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


@contextmanager
def timed_section(bucket: Dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        bucket[key] = (time.perf_counter() - start) * 1000.0
