"""LaMa：在 Demo 本进程内推理（simple-lama-inpainting）。"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from .roi import paste_back, resize_for_infer


def pick_device() -> torch.device:
    force = os.environ.get("LAMA_DEVICE", "cpu").lower()
    if force in {"cpu", "cuda", "mps"}:
        if force == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if force == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return torch.device(force)
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_lama():
    from simple_lama_inpainting import SimpleLama

    device = pick_device()
    print(f"[lama] loading LaMa on {device} ...")
    return SimpleLama(device=device)


def run_lama(image: Image.Image, mask: Image.Image) -> Image.Image:
    from simple_lama_inpainting.utils.util import prepare_img_and_mask

    lama = get_lama()
    orig_w, orig_h = image.size
    img_t, mask_t = prepare_img_and_mask(image, mask, lama.device, pad_out_to_modulo=8)

    t0 = time.perf_counter()
    with torch.inference_mode():
        inpainted = lama.model(img_t, mask_t)
        if lama.device.type == "cuda":
            torch.cuda.synchronize()
        out = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
    print(f"[lama] gpu_infer {time.perf_counter() - t0:.1f}s tensor={tuple(img_t.shape)} device={lama.device}")

    out = out[:orig_h, :orig_w]
    return Image.fromarray(out, mode="RGB")


class LamaBackend:
    name = "LaMa"

    def inpaint(self, image: Image.Image, mask: Image.Image, max_side: int) -> Image.Image:
        infer_img, infer_mask, scale = resize_for_infer(image, mask, int(max_side))
        print(f"[lama] original={image.size} infer={infer_img.size} scale={scale:.3f} device={pick_device()}")
        inpainted = run_lama(infer_img, infer_mask)
        t0 = time.perf_counter()
        out = paste_back(image, inpainted, mask)
        print(f"[lama] paste_back {time.perf_counter() - t0:.1f}s {infer_img.size} → {image.size}")
        return out

    def warmup(self) -> None:
        get_lama()
