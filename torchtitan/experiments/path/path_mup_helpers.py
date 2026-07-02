from __future__ import annotations

BASE_WIDTH = 256


def hidden_std(fan_in: int) -> float:
    return fan_in**-0.5


def scale_dims(dims_base: tuple[int, ...], width: int, base_width: int = BASE_WIDTH) -> tuple[int, ...]:
    return tuple(d * width // base_width for d in dims_base)
