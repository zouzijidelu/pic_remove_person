"""Mask ROI 裁切与贴回。扩散模型只修人像附近一块，避免整幅全景爆显存。"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def mask_bbox(mask: Image.Image, margin: int) -> tuple[int, int, int, int] | None:
    arr = np.array(mask.convert("L"))
    ys, xs = np.where(arr > 127)
    if len(xs) == 0:
        return None
    h, w = arr.shape[:2]
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(w, int(xs.max()) + 1 + margin)
    y2 = min(h, int(ys.max()) + 1 + margin)
    return x1, y1, x2, y2


def resize_max_side(img: Image.Image, mask: Image.Image, max_side: int):
    w, h = img.size
    long_side = max(w, h)
    if max_side <= 0 or long_side <= max_side:
        return img, mask, 1.0
    scale = max_side / float(long_side)
    nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    nw -= nw % 8
    nh -= nh % 8
    nw, nh = max(8, nw), max(8, nh)
    return (
        img.resize((nw, nh), Image.Resampling.LANCZOS),
        mask.resize((nw, nh), Image.Resampling.NEAREST),
        scale,
    )


def resize_for_infer(image: Image.Image, mask: Image.Image, max_side: int):
    """整图降采样（LaMa）。边长对齐到 8。"""
    return resize_max_side(image, mask, max_side)


def crop_roi(
    image: Image.Image,
    mask: Image.Image,
    margin: int,
    max_side: int,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int], float]:
    bbox = mask_bbox(mask, margin)
    if bbox is None:
        raise ValueError("Mask 为空，无法裁 ROI")
    x1, y1, x2, y2 = bbox
    crop_img = image.crop((x1, y1, x2, y2))
    crop_mask = mask.crop((x1, y1, x2, y2))
    crop_img, crop_mask, scale = resize_max_side(crop_img, crop_mask, max_side)
    return crop_img, crop_mask, bbox, scale


def paste_back(original: Image.Image, inpainted_small: Image.Image, mask_full: Image.Image) -> Image.Image:
    """把降采样修复结果仅在 Mask 区域羽化贴回原图。"""
    if inpainted_small.size == original.size:
        return inpainted_small

    up = inpainted_small.resize(original.size, Image.Resampling.LANCZOS)
    orig = np.array(original.convert("RGB"))
    up_arr = np.array(up.convert("RGB"))
    m = np.array(mask_full.resize(original.size, Image.Resampling.NEAREST))
    m_f = (m.astype(np.float32) / 255.0)[..., None]
    if m_f.max() > 0:
        m_blur = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=2)
        m_f = (m_blur / 255.0)[..., None]
    blended = orig.astype(np.float32) * (1.0 - m_f) + up_arr.astype(np.float32) * m_f
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def match_inpaint_to_surroundings(
    orig_rgb: np.ndarray,
    inpaint_rgb: np.ndarray,
    mask_u8: np.ndarray,
    ring_px: int = 28,
) -> np.ndarray:
    """把修复区域的亮度/色度对齐到 Mask 外围一圈真实背景。

    PowerPaint/SD 的 VAE 常把整块画亮；全局直方图对齐用不了周围墙地，这里按邻域做 Lab 校正。
    """
    hole = mask_u8 > 127
    if hole.sum() < 30:
        return inpaint_rgb

    k = max(5, int(ring_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(hole.astype(np.uint8) * 255, kernel)
    ring = (dil > 127) & (~hole)
    if ring.sum() < 30:
        return inpaint_rgb

    orig_lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    inp_lab = cv2.cvtColor(inpaint_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out_lab = inp_lab.copy()
    for c in range(3):
        src = inp_lab[..., c][hole]
        ref = orig_lab[..., c][ring]
        src_mean = float(src.mean())
        ref_mean = float(ref.mean())
        if c == 0:
            src_std = float(src.std()) + 1e-5
            ref_std = float(ref.std()) + 1e-5
            scale = float(np.clip(ref_std / src_std, 0.7, 1.3))
            out_lab[..., c][hole] = (src - src_mean) * scale + ref_mean
        else:
            out_lab[..., c][hole] = src + (ref_mean - src_mean)
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return np.where(hole[..., None], out, orig_rgb)


def paste_roi(
    original: Image.Image,
    roi_result: Image.Image,
    bbox: tuple[int, int, int, int],
    mask_full: Image.Image,
    color_match: bool = True,
) -> Image.Image:
    """把 ROI 修复块贴回全图，仅 Mask 区域羽化混合。"""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return original

    patch = roi_result.resize((bw, bh), Image.Resampling.LANCZOS)
    orig = np.array(original.convert("RGB"))
    patch_arr = np.array(patch.convert("RGB"))
    m = np.array(mask_full.convert("L").resize(original.size, Image.Resampling.NEAREST))
    m_roi = m[y1:y2, x1:x2]
    orig_roi = orig[y1:y2, x1:x2]
    if color_match and m_roi.max() > 0:
        ring = max(16, int(round(min(bw, bh) * 0.04)))
        patch_arr = match_inpaint_to_surroundings(orig_roi, patch_arr, m_roi, ring_px=ring)
    if m_roi.max() > 0:
        sigma = max(3.0, min(bw, bh) * 0.008)
        m_blur = cv2.GaussianBlur(m_roi.astype(np.float32), (0, 0), sigmaX=sigma)
        m_f = (m_blur / 255.0)[..., None]
    else:
        m_f = (m_roi.astype(np.float32) / 255.0)[..., None]
    blended = orig_roi.astype(np.float32) * (1.0 - m_f) + patch_arr.astype(np.float32) * m_f
    orig[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(orig, mode="RGB")
