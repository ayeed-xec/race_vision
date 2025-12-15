"""Lightweight numpy compatibility shim for offline testing.
This is **not** a full numpy implementation; it only supports the small
subset of APIs exercised by the tests in this repository. The goal is to
avoid a hard dependency on the real numpy package in CI environments
without network access.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

uint8 = int
float32 = float
float64 = float

def _build_shape(data: Any) -> Tuple[int, ...]:
    if isinstance(data, ndarray):
        return data.shape
    if not isinstance(data, (list, tuple)):
        return ()
    if len(data) == 0:
        return (0,)
    return (len(data),) + _build_shape(data[0])


def _fill_zeros(shape: Sequence[int]) -> Any:
    if len(shape) == 0:
        return 0
    return [_fill_zeros(shape[1:]) for _ in range(int(shape[0]))]


class ndarray:
    def __init__(self, data: Any, dtype: Any | None = None, shape: Tuple[int, ...] | None = None):
        self._data = data
        self.dtype = dtype or float
        self.shape = shape or _build_shape(data)

    def astype(self, dtype: Any):
        def _convert(value: Any):
            try:
                return dtype(value)  # type: ignore[arg-type]
            except Exception:
                return value

        def _apply(obj: Any):
            if isinstance(obj, list):
                return [_apply(v) for v in obj]
            return _convert(obj)

        return ndarray(_apply(self._data), dtype=dtype, shape=self.shape)

    def mean(self) -> float:
        flat = list(self._flatten())
        return float(sum(flat) / len(flat)) if flat else 0.0

    def _flatten(self) -> Iterable[float]:
        def _yield(obj: Any):
            if isinstance(obj, list):
                for v in obj:
                    yield from _yield(v)
            else:
                yield float(obj)

        return list(_yield(self._data))

    def __sub__(self, other: "ndarray") -> "ndarray":
        return ndarray(_binary_op(self._data, other._data, lambda a, b: a - b), dtype=self.dtype, shape=self.shape)

    def __add__(self, other: "ndarray") -> "ndarray":
        return ndarray(_binary_op(self._data, other._data, lambda a, b: a + b), dtype=self.dtype, shape=self.shape)

    def __abs__(self) -> "ndarray":
        def _abs(val: Any):
            if isinstance(val, list):
                return [_abs(v) for v in val]
            return abs(val)

        return ndarray(_abs(self._data), dtype=self.dtype, shape=self.shape)

    def __truediv__(self, other: float) -> "ndarray":
        def _div(val: Any):
            if isinstance(val, list):
                return [_div(v) for v in val]
            return val / other

        return ndarray(_div(self._data), dtype=self.dtype, shape=self.shape)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]


def _binary_op(a: Any, b: Any, fn):
    if isinstance(a, list) and isinstance(b, list):
        return [_binary_op(x, y, fn) for x, y in zip(a, b)]
    return fn(a, b)


def zeros(shape: Sequence[int], dtype: Any | None = None) -> ndarray:
    data = _fill_zeros(shape)
    return ndarray(data, dtype=dtype or float, shape=tuple(int(s) for s in shape))


def stack(arrays: Sequence[ndarray], axis: int = 0) -> ndarray:
    if not arrays:
        return ndarray([], dtype=float, shape=(0,))
    data: List[Any] = [a._data for a in arrays]
    if axis == -1:
        stacked = []
        for items in zip(*data):
            stacked.append(list(items))
        shape = arrays[0].shape + (len(arrays),)
        return ndarray(stacked, dtype=arrays[0].dtype, shape=shape)
    return ndarray(data, dtype=arrays[0].dtype, shape=(len(arrays),) + arrays[0].shape)


def abs_(obj: ndarray) -> ndarray:
    return obj.__abs__()


def array(data: Any, dtype: Any | None = None) -> ndarray:
    return ndarray(data, dtype=dtype or float)

__all__ = [
    "ndarray",
    "zeros",
    "stack",
    "array",
    "abs_",
    "uint8",
    "float32",
    "float64",
]
