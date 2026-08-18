"""PowerPaint：经旁路 iopaint-bench 进程推理，不装进 Demo 的 .venv。"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from pathlib import Path

from PIL import Image

from .roi import crop_roi, paste_roi

ROOT = Path(__file__).resolve().parents[1]
BENCH_DEFAULT = ROOT.parent / "iopaint-bench"
MODEL_ID = "Sanster/PowerPaint-V1-stable-diffusion-inpainting"
READY_TIMEOUT = int(os.environ.get("POWERPAINT_READY_TIMEOUT", "600"))
INFER_TIMEOUT = int(os.environ.get("POWERPAINT_INFER_TIMEOUT", "300"))


def bench_root() -> Path:
    return Path(os.environ.get("IOPAINT_BENCH", BENCH_DEFAULT)).resolve()


def pick_device() -> str:
    force = os.environ.get("POWERPAINT_DEVICE", "auto").strip().lower()
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
    if not py.exists():
        raise FileNotFoundError(
            "未找到 iopaint-bench 虚拟环境。请先：cd ../iopaint-bench && ./setup.sh"
        )
    return py


class PowerPaintBackend:
    name = "PowerPaint"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._tmp = ROOT / "output" / ".pp_tmp"
        self._tmp.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin:
                proc.stdin.write('{"cmd":"quit"}\n')
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def _read_json(self, timeout: int) -> dict:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("PowerPaint worker 未启动")
        line = ""

        def _read() -> None:
            nonlocal line
            line = proc.stdout.readline()

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"等待 PowerPaint worker 超时（{timeout}s）")
        if not line:
            code = proc.poll()
            raise RuntimeError(f"PowerPaint worker 已退出（code={code}），请看启动 Demo 的终端日志")
        return json.loads(line)

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        bench = bench_root()
        worker = bench / "inpaint_worker.py"
        config = bench / "configs" / "powerpaint_fast.json"
        if not worker.exists():
            raise FileNotFoundError(f"缺少 worker 脚本: {worker}")
        if not config.exists():
            raise FileNotFoundError(f"缺少 PowerPaint 配置: {config}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["HF_HOME"] = str(bench / ".cache" / "huggingface")
        env["HUGGINGFACE_HUB_CACHE"] = str(bench / ".cache" / "huggingface" / "hub")
        env.pop("TORCH_HOME", None)

        device = pick_device()
        print(f"[powerpaint] starting worker model={MODEL_ID} device={device} bench={bench}")
        self._proc = subprocess.Popen(
            [
                str(_python()),
                str(worker),
                "--model",
                MODEL_ID,
                "--device",
                device,
                "--config",
                str(config),
            ],
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
            raise RuntimeError(f"PowerPaint 加载失败: {msg.get('error', msg)}")
        print(f"[powerpaint] worker ready: {msg}")

    def _call_worker(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        img_p = self._tmp / "roi.png"
        mask_p = self._tmp / "mask.png"
        out_p = self._tmp / "out.png"
        if out_p.exists():
            out_p.unlink()
        image.save(img_p)
        mask.save(mask_p)

        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(
            json.dumps(
                {
                    "cmd": "inpaint",
                    "image": str(img_p),
                    "mask": str(mask_p),
                    "output": str(out_p),
                    "max_side": 0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._proc.stdin.flush()
        msg = self._read_json(INFER_TIMEOUT)
        if not msg.get("ok"):
            raise RuntimeError(f"PowerPaint 推理失败: {msg.get('error', msg)}")
        if not out_p.exists():
            raise RuntimeError("PowerPaint 未写出结果图")
        return Image.open(out_p).convert("RGB")

    def inpaint(self, image: Image.Image, mask: Image.Image, max_side: int) -> Image.Image:
        margin = int(os.environ.get("POWERPAINT_MARGIN", "256"))
        roi_img, roi_mask, bbox, scale = crop_roi(image, mask, margin=margin, max_side=int(max_side))
        print(
            f"[powerpaint] original={image.size} roi={roi_img.size} "
            f"bbox={bbox} scale={scale:.3f} device={pick_device()}"
        )
        with self._lock:
            try:
                self._ensure_worker()
                roi_out = self._call_worker(roi_img, roi_mask)
            except Exception:
                self.close()
                raise
        return paste_roi(image, roi_out, bbox, mask)
