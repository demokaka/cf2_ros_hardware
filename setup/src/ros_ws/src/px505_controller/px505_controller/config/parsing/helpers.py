from __future__ import annotations
from typing import Any, Mapping


def require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping/dict, got {type(value).__name__}")
    return value


def require_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a string, got {type(value).__name__}")
    return value


def require_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{where} must be an int, got bool")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{where} must be an int, got {value!r}")


def require_float(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{where} must be a number, got bool")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{where} must be a number, got {value!r}")


def require_vector(value: Any, where: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{where} must be a list/tuple of numbers, got {type(value).__name__}")
    if len(value) == 0:
        raise ValueError(f"{where} must not be empty")
    out: list[float] = []
    for i, x in enumerate(value):
        out.append(require_float(x, f"{where}[{i}]"))
    return tuple(out)
