"""Qwen-Image-Edit-2511：经旁路 qwen-edit-bench 进程推理，不装进 Demo 的 .venv。"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image

from .roi import crop_roi, paste_roi

ROOT = Path(__file__).resolve().parents[1]
BENCH_DEFAULT = ROOT.parent / "qwen-edit-bench"
READY_TIMEOUT = int(os.environ.get("QWEN_EDIT_READY_TIMEOUT", "1800"))
INFER_TIMEOUT = int(os.environ.get("QWEN_EDIT_INFER_TIMEOUT", "600"))
DEFAULT_PROMPT = (
    "Remove the person. Reconstruct the background only. Do not add new people."
)


def bench_root() -> Path:
    return Path(os.environ.get("QWEN_EDIT_BENCH", BENCH_DEFAULT)).resolve()


def pick_device() -> str:
    force = os.environ.get("QWEN_EDIT_DEVICE", "auto").strip().lower()
    if force in {"cpu", "cuda", "mps"}:
        return force
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _python() -> Path:
    py = bench_root() / ".venv" / "bin" / "python"
    if py.exists():
        return py
    if _dry_run():
        demo_py = ROOT / ".venv" / "bin" / "python"
        if demo_py.exists():
            return demo_py
        return Path(sys.executable)
    raise FileNotFoundError(
        "未找到 qwen-edit-bench 虚拟环境。请先：cd ../qwen-edit-bench && ./setup.sh"
    )


def _dry_run() -> bool:
    return os.environ.get("QWEN_EDIT_DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "on"}


class QwenEditBackend:
    name = "Qwen-Edit"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._tmp = ROOT / "output" / ".qwen_tmp"
        self._tmp.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write('{"cmd":"quit"}\n')
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=3)
                except Exception:
                    pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _read_json(self, timeout: int) -> dict:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("Qwen-Edit worker 未启动")
        line = ""

        def _read() -> None:
            nonlocal line
            line = proc.stdout.readline()

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"等待 Qwen-Edit worker 超时（{timeout}s）")
        if not line:
            code = proc.poll()
            raise RuntimeError(f"Qwen-Edit worker 已退出（code={code}），请看启动 Demo 的终端日志")
        return json.loads(line)

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        from . import release_other_heavy_backends

        release_other_heavy_backends(self.name)

        bench = bench_root()
        worker = bench / "edit_worker.py"
        config = bench / "configs" / "qwen_fast.json"
        if not worker.exists():
            raise FileNotFoundError(f"缺少 worker 脚本: {worker}")
        if not config.exists():
            raise FileNotFoundError(f"缺少 Qwen-Edit 配置: {config}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        env["HF_HOME"] = str(bench / ".cache" / "huggingface")
        env["HUGGINGFACE_HUB_CACHE"] = str(bench / ".cache" / "huggingface" / "hub")
        env.pop("TORCH_HOME", None)

        device = pick_device()
        cmd = [str(_python()), str(worker), "--device", device, "--config", str(config)]
        if _dry_run():
            cmd.append("--dry-run")
        print(f"[qwen-edit] starting worker device={device} dry_run={_dry_run()} bench={bench}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(bench),
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        msg = self._read_json(READY_TIMEOUT)
        if not msg.get("ok"):
            self.close()
            raise RuntimeError(f"Qwen-Edit 加载失败: {msg.get('error', msg)}")
        print(f"[qwen-edit] worker ready: {msg}")

    def _call_worker(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        img_p = self._tmp / "roi.png"
        mask_p = self._tmp / "mask.png"
        out_p = self._tmp / "out.png"
        if out_p.exists():
            out_p.unlink()
        image.save(img_p)
        mask.save(mask_p)
        prompt = os.environ.get("QWEN_EDIT_PROMPT", DEFAULT_PROMPT).strip() or DEFAULT_PROMPT

        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(
            json.dumps(
                {
                    "cmd": "inpaint",
                    "image": str(img_p),
                    "mask": str(mask_p),
                    "output": str(out_p),
                    "prompt": prompt,
                    "max_side": 0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._proc.stdin.flush()
        msg = self._read_json(INFER_TIMEOUT)
        if not msg.get("ok"):
            raise RuntimeError(f"Qwen-Edit 推理失败: {msg.get('error', msg)}")
        if not out_p.exists():
            raise RuntimeError("Qwen-Edit 未写出结果图")
        return Image.open(out_p).convert("RGB")

    def inpaint(self, image: Image.Image, mask: Image.Image, max_side: int) -> Image.Image:
        margin = int(os.environ.get("QWEN_EDIT_MARGIN", "256"))
        t0 = time.perf_counter()
        roi_img, roi_mask, bbox, scale = crop_roi(image, mask, margin=margin, max_side=int(max_side))
        crop_s = time.perf_counter() - t0
        print(
            f"[qwen-edit] original={image.size} roi={roi_img.size} "
            f"bbox={bbox} scale={scale:.3f} device={pick_device()} crop={crop_s:.2f}s"
        )
        with self._lock:
            try:
                t1 = time.perf_counter()
                self._ensure_worker()
                ready_s = time.perf_counter() - t1
                t2 = time.perf_counter()
                roi_out = self._call_worker(roi_img, roi_mask)
                worker_s = time.perf_counter() - t2
            except Exception:
                self.close()
                raise
        t3 = time.perf_counter()
        out = paste_roi(image, roi_out, bbox, mask)
        print(
            f"[qwen-edit] ensure_worker={ready_s:.2f}s worker={worker_s:.2f}s "
            f"paste_roi={time.perf_counter() - t3:.2f}s"
        )
        return out

    def warmup(self) -> None:
        if os.environ.get("WARMUP_QWEN", "0").strip() != "1":
            print("[qwen-edit] skip warmup（WARMUP_QWEN!=1，首次使用时再加载）")
            return
        with self._lock:
            self._ensure_worker()
