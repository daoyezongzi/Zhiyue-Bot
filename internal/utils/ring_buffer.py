from __future__ import annotations

from collections import deque
from typing import Deque, Generic, Iterable, TypeVar


T = TypeVar("T")


class RingBuffer(Generic[T]):
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self._buffer: Deque[T] = deque(maxlen=capacity)

    def push(self, value: T) -> None:
        self._buffer.append(value)

    def get_all(self) -> list[T]:
        return list(self._buffer)

    def extend(self, values: Iterable[T]) -> None:
        self._buffer.extend(values)

    def is_empty(self) -> bool:
        return not self._buffer
