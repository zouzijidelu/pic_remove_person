"""一次消除的耗时与显存快照。日志给模型对比 / 选默认 / 后续优化用。"""

from __future__ import annotations

import os
import resource
import subprocess
import time


def rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux：KB；macOS：字节
    if rss > 10_000_000:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def torch_mem() -> dict:
    out = {"torch_alloc_mb": 0.0, "torch_reserved_mb": 0.0, "torch_peak_mb": 0.0}
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        torch.cuda.synchronize()
        out["torch_alloc_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        out["torch_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        out["torch_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return out


def reset_torch_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def nvidia_smi() -> dict:
    info: dict = {
        "smi_used_mb": None,
        "smi_total_mb": None,
        "smi_util": None,
        "procs": [],
    }
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        used, total, util = [x.strip() for x in raw.strip().split(",")[:3]]
        info["smi_used_mb"] = int(float(used))
        info["smi_total_mb"] = int(float(total))
        info["smi_util"] = int(float(util))
    except Exception:
        pass
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
        procs = []
        for line in rows.splitlines() if rows else []:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = os.path.basename(parts[2])
                procs.append(f"{name}:{parts[0]}={parts[1]}MB")
        info["procs"] = procs
    except Exception:
        pass
    return info


class StageClock:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.mark = self.t0
        self.stages: list[tuple[str, float]] = []

    def tick(self, name: str) -> float:
        now = time.perf_counter()
        dt = now - self.mark
        self.stages.append((name, dt))
        self.mark = now
        return dt

    def total(self) -> float:
        return time.perf_counter() - self.t0

    def summary(self) -> str:
        parts = [f"{n}={d:.2f}s" for n, d in self.stages]
        return " ".join(parts) + f" handler={self.total():.2f}s"


def format_resources(torch_info: dict | None = None, smi: dict | None = None) -> str:
    torch_info = torch_info or torch_mem()
    smi = smi or nvidia_smi()
    bits = [
        f"rss={rss_mb():.0f}MB",
        f"torch_alloc={torch_info['torch_alloc_mb']:.0f}",
        f"peak={torch_info['torch_peak_mb']:.0f}",
        f"reserved={torch_info['torch_reserved_mb']:.0f}MB",
    ]
    if smi.get("smi_used_mb") is not None:
        bits.append(f"smi={smi['smi_used_mb']}/{smi['smi_total_mb']}MB")
        bits.append(f"util={smi['smi_util']}%")
    if smi.get("procs"):
        bits.append("procs=" + ",".join(smi["procs"]))
    return " ".join(bits)
