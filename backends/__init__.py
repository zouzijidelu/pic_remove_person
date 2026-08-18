"""修复模型注册表。换模型只改这里 + 对应 backend，不碰 SAM / 笔刷。"""

from __future__ import annotations

import os

from .lama import LamaBackend

BACKENDS: dict[str, type] = {
    "LaMa": LamaBackend,
}


def _want_powerpaint() -> bool:
    flag = os.environ.get("ENABLE_POWERPAINT", "auto").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    from .powerpaint import bench_root

    return (bench_root() / ".venv" / "bin" / "python").exists()


if _want_powerpaint():
    from .powerpaint import PowerPaintBackend

    BACKENDS["PowerPaint"] = PowerPaintBackend

_INSTANCES: dict[str, object] = {}


def list_backend_names() -> list[str]:
    return list(BACKENDS.keys())


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"未知修复模型: {name}，可选 {list_backend_names()}")
    if name not in _INSTANCES:
        _INSTANCES[name] = BACKENDS[name]()
    return _INSTANCES[name]
