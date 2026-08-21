"""修复模型注册表。换模型只改这里 + 对应 backend，不碰 SAM / 笔刷。"""

from __future__ import annotations

import os

from .lama import LamaBackend

BACKENDS: dict[str, type] = {
    "LaMa": LamaBackend,
}

HEAVY_BACKENDS = ("PowerPaint", "Qwen-Edit")


def _flag_on(name: str, default: str = "auto") -> str:
    return os.environ.get(name, default).strip().lower()


def _want_powerpaint() -> bool:
    flag = _flag_on("ENABLE_POWERPAINT")
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    from .powerpaint import bench_root

    return (bench_root() / ".venv" / "bin" / "python").exists()


def _want_qwen_edit() -> bool:
    flag = _flag_on("ENABLE_QWEN_EDIT")
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    from .qwen_edit import bench_root

    bench = bench_root()
    if (bench / ".venv" / "bin" / "python").exists():
        return True
    dry = os.environ.get("QWEN_EDIT_DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "on"}
    return dry and (bench / "edit_worker.py").exists()


if _want_powerpaint():
    from .powerpaint import PowerPaintBackend

    BACKENDS["PowerPaint"] = PowerPaintBackend

if _want_qwen_edit():
    from .qwen_edit import QwenEditBackend

    BACKENDS["Qwen-Edit"] = QwenEditBackend

_INSTANCES: dict[str, object] = {}


def list_backend_names() -> list[str]:
    return list(BACKENDS.keys())


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"未知修复模型: {name}，可选 {list_backend_names()}")
    if name not in _INSTANCES:
        _INSTANCES[name] = BACKENDS[name]()
    return _INSTANCES[name]


def release_other_heavy_backends(keep: str) -> None:
    """PowerPaint 与 Qwen 互斥常驻，用谁加载谁。"""
    for name in HEAVY_BACKENDS:
        if name == keep:
            continue
        inst = _INSTANCES.get(name)
        if inst is None:
            continue
        close = getattr(inst, "close", None)
        if not callable(close):
            continue
        print(f"[backends] 释放 {name}，避免与 {keep} 同时占显存")
        close()
        _INSTANCES.pop(name, None)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from .profile import nvidia_smi, format_resources

        smi = nvidia_smi()
        print(f"[backends] 释放后 GPU {format_resources(smi=smi)}")
    except Exception:
        pass
