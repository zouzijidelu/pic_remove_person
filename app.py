"""全景人像消除 Demo：SAM 点选 / 矩形框粗 Mask + 笔刷精修 + 可切换修复模型。"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from backends import get_backend, list_backend_names
from backends.profile import StageClock, format_resources, reset_torch_peak

ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = ROOT / "resource"
RESOURCE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
SAM_CKPT = ROOT / ".cache" / "sam" / "sam_vit_b_01ec64.pth"
SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# 推理最长边上限（过大易被 macOS 因内存杀掉）
MAX_INFER_SIDE = int(os.environ.get("MAX_INFER_SIDE", "1536"))
POWERPAINT_MAX_SIDE = int(os.environ.get("POWERPAINT_MAX_SIDE", "768"))
QWEN_EDIT_MAX_SIDE = int(os.environ.get("QWEN_EDIT_MAX_SIDE", "768"))
# SAM 编码用图最长边（越小越快；点坐标会映射回原图）
SAM_MAX_SIDE = int(os.environ.get("SAM_MAX_SIDE", "1024"))
PREVIEW_MAX_SIDE = int(os.environ.get("PREVIEW_MAX_SIDE", "960"))
DEFAULT_MODEL = os.environ.get("INPAINT_MODEL", "LaMa")

# 原尺寸 Mask 只放服务端内存，浏览器只拿 session id + 缩略预览
_MASK_LOCK = threading.Lock()
_MASK_STORE: OrderedDict[str, dict] = OrderedDict()
_MAX_MASK_SESSIONS = 32


def _ensure_session_id(sid: str | None) -> str:
    return sid if sid else uuid.uuid4().hex


def _get_masks(sid: str | None) -> tuple[Image.Image | None, Image.Image | None]:
    if not sid:
        return None, None
    with _MASK_LOCK:
        slot = _MASK_STORE.get(sid) or {}
        return slot.get("combined"), slot.get("rect")


def _put_masks(sid: str, combined: Image.Image | None, rect: Image.Image | None) -> None:
    with _MASK_LOCK:
        while sid not in _MASK_STORE and len(_MASK_STORE) >= _MAX_MASK_SESSIONS:
            _MASK_STORE.popitem(last=False)
        _MASK_STORE[sid] = {"combined": combined, "rect": rect}
        _MASK_STORE.move_to_end(sid)


def _reset_masks(sid: str) -> None:
    _put_masks(sid, None, None)


def list_resource_examples() -> list[list[str]]:
    if not RESOURCE_DIR.is_dir():
        return []
    files = sorted(
        p for p in RESOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in RESOURCE_EXTS
    )
    return [[str(p)] for p in files]


def _safe_stem(name: str) -> str:
    stem = Path(str(name)).stem
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip("._")
    return (stem or "upload")[:80]


def _image_stem(img) -> str:
    """从上传/样例图尽量取出原文件名（不含扩展名）。"""
    raw = None
    if img is None:
        return "upload"
    if isinstance(img, Image.Image):
        raw = getattr(img, "filename", None)
    elif isinstance(img, (str, Path)):
        raw = str(img)
    elif isinstance(img, dict):
        raw = img.get("path") or img.get("orig_name") or img.get("name")
    if not raw:
        return "upload"
    return _safe_stem(raw)


def _output_paths(source_stem: str, model_name: str) -> tuple[Path, Path]:
    stem = _safe_stem(source_stem or "upload")
    model = _safe_stem(model_name or "model")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{stem}_{model}_{ts}"
    return OUTPUT_DIR / f"{prefix}.jpg", OUTPUT_DIR / f"{prefix}_mask.png"


def pick_sam_device() -> torch.device:
    # SAM 在 MPS 上通常可用；也可强制 LAMA_DEVICE / SAM_DEVICE
    force = os.environ.get("SAM_DEVICE", os.environ.get("LAMA_DEVICE", "cpu")).lower()
    if force == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if force == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_sam_checkpoint() -> Path:
    SAM_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if SAM_CKPT.exists() and SAM_CKPT.stat().st_size > 300_000_000:
        return SAM_CKPT
    import urllib.request

    print(f"[sam] downloading checkpoint → {SAM_CKPT}")
    tmp = SAM_CKPT.with_suffix(".pth.partial")
    urllib.request.urlretrieve(SAM_CKPT_URL, tmp)
    tmp.replace(SAM_CKPT)
    return SAM_CKPT


@lru_cache(maxsize=1)
def get_sam_predictor():
    from segment_anything import SamPredictor, sam_model_registry

    ckpt = ensure_sam_checkpoint()
    device = pick_sam_device()
    print(f"[sam] loading ViT-B on {device} ...")
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def _to_pil_rgb(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, (str, Path)):
        p = Path(img)
        if not p.exists():
            return None
        return Image.open(p).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    return Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")


def _resize_max_side(pil: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    w, h = pil.size
    long_side = max(w, h)
    if long_side <= max_side:
        return pil, 1.0
    scale = max_side / float(long_side)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return pil.resize((nw, nh), Image.Resampling.LANCZOS), scale


def _preview_size(pil: Image.Image) -> Image.Image:
    preview, _ = _resize_max_side(pil, PREVIEW_MAX_SIDE)
    return preview


def _extract_brush_mask(editor_value, target_size: tuple[int, int]) -> Image.Image | None:
    """从 ImageEditor 提取笔刷 Mask，并缩放到 target_size=(W,H)。"""
    if editor_value is None:
        return None

    layers = editor_value.get("layers") or []
    background = editor_value.get("background")
    composite = editor_value.get("composite")

    tw, th = target_size
    mask_arr = np.zeros((th, tw), dtype=np.uint8)

    if layers:
        for layer in layers:
            if layer is None:
                continue
            layer_img = layer if isinstance(layer, Image.Image) else Image.fromarray(np.array(layer))
            layer_img = layer_img.convert("RGBA")
            alpha = np.array(layer_img.split()[-1])
            if alpha.max() == 0:
                rgb = np.array(layer_img.convert("RGB"))
                painted = (rgb.sum(axis=2) > 15).astype(np.uint8) * 255
            else:
                painted = (alpha > 10).astype(np.uint8) * 255
            if painted.shape[:2] != (th, tw):
                painted = cv2.resize(painted, (tw, th), interpolation=cv2.INTER_NEAREST)
            mask_arr = np.maximum(mask_arr, painted)
    elif composite is not None and background is not None:
        bg = np.array(_to_pil_rgb(background).resize((tw, th), Image.Resampling.BILINEAR))
        comp = np.array(_to_pil_rgb(composite).resize((tw, th), Image.Resampling.BILINEAR))
        diff = np.abs(comp.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
        mask_arr = ((diff > 20) * 255).astype(np.uint8)
    else:
        return None

    if mask_arr.max() == 0:
        return None
    return Image.fromarray(mask_arr, mode="L")


def dilate_mask(mask: Image.Image, pixels: int) -> Image.Image:
    if pixels <= 0:
        return mask
    arr = np.array(mask)
    k = max(1, int(pixels) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    arr = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(arr, mode="L")


def mask_to_editor(background: Image.Image, mask: Image.Image, alpha: int = 150):
    """把 Mask 画成红色半透明图层，放进 ImageEditor 供继续精修。"""
    bg = _preview_size(background).convert("RGBA")
    mw, mh = bg.size
    m = np.array(mask.resize((mw, mh), Image.Resampling.NEAREST))
    layer = np.zeros((mh, mw, 4), dtype=np.uint8)
    hit = m > 127
    layer[hit] = [255, 0, 0, int(alpha)]
    layer_img = Image.fromarray(layer, mode="RGBA")
    composite = Image.alpha_composite(bg, layer_img)
    return {
        "background": bg,
        "layers": [layer_img],
        "composite": composite,
    }


def overlay_mask_preview(
    image: Image.Image,
    mask: Image.Image | None,
    points: list,
    rects: list | None = None,
    pending_corner: tuple[int, int] | None = None,
) -> Image.Image:
    """在预览图上叠 Mask + SAM 点 + 矩形框。"""
    preview = _preview_size(image).convert("RGB")
    arr = np.array(preview).astype(np.float32)
    if mask is not None:
        m = np.array(mask.resize(preview.size, Image.Resampling.NEAREST)) > 127
        arr[m] = arr[m] * 0.45 + np.array([255, 60, 60], dtype=np.float32) * 0.55
    out = np.clip(arr, 0, 255).astype(np.uint8)
    draw_arr = out
    ow, oh = image.size
    pw, ph = preview.size
    sx, sy = pw / float(ow), ph / float(oh)

    for x, y, label in points or []:
        px, py = int(x * sx), int(y * sy)
        color = (0, 220, 0) if label == 1 else (40, 40, 255)
        cv2.circle(draw_arr, (px, py), 8, color, -1)
        cv2.circle(draw_arr, (px, py), 10, (255, 255, 255), 2)

    for x1, y1, x2, y2 in rects or []:
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(draw_arr, p1, p2, (0, 255, 255), 3)

    if pending_corner is not None:
        px, py = int(pending_corner[0] * sx), int(pending_corner[1] * sy)
        cv2.circle(draw_arr, (px, py), 14, (0, 255, 255), 3)
        cv2.drawMarker(
            draw_arr, (px, py), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2
        )
        cv2.putText(
            draw_arr,
            "1",
            (px + 16, py - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(draw_arr, mode="RGB")


def _preview_to_original_xy(original: Image.Image, evt_index) -> tuple[int, int]:
    preview = _preview_size(original)
    pw, ph = preview.size
    ow, oh = original.size
    cx, cy = evt_index
    ox = int(np.clip(cx * ow / pw, 0, ow - 1))
    oy = int(np.clip(cy * oh / ph, 0, oh - 1))
    return ox, oy


def _normalize_rect(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def union_rect_mask(mask: Image.Image | None, size: tuple[int, int], rect: tuple[int, int, int, int]) -> Image.Image:
    """把矩形区域并入 Mask（原图坐标）。"""
    w, h = size
    if mask is None:
        arr = np.zeros((h, w), dtype=np.uint8)
    else:
        arr = np.array(mask.convert("L").resize((w, h), Image.Resampling.NEAREST))
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 > x1 and y2 > y1:
        arr[y1 : y2 + 1, x1 : x2 + 1] = 255
    return Image.fromarray(arr, mode="L")


_SAM_EMBED_KEY: tuple | None = None


def _ensure_sam_embedding(original_img, original: Image.Image) -> float:
    """同一张原图多次点选时复用 SAM 图像编码，避免每次 set_image。"""
    global _SAM_EMBED_KEY
    predictor = get_sam_predictor()
    sam_img, scale = _resize_max_side(original, SAM_MAX_SIDE)
    key: tuple | None = None
    if isinstance(original_img, (str, Path)):
        p = Path(original_img)
        try:
            st = p.stat()
            key = (str(p.resolve()), st.st_mtime_ns, st.st_size, sam_img.size)
        except OSError:
            key = None
    if key is None or key != _SAM_EMBED_KEY:
        predictor.set_image(np.array(sam_img.convert("RGB")))
        _SAM_EMBED_KEY = key
    return scale


def run_sam_on_points(original_img, original: Image.Image, points: list[tuple[int, int, int]]) -> Image.Image:
    """points: [(x,y,label)] 原图像素坐标，label 1=正点 0=负点。"""
    if not points:
        raise gr.Error("请先在图上点击人像（可多点）")

    predictor = get_sam_predictor()
    scale = _ensure_sam_embedding(original_img, original)

    coords = np.array([[p[0] * scale, p[1] * scale] for p in points], dtype=np.float32)
    labels = np.array([p[2] for p in points], dtype=np.int32)

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
    best = masks[int(np.argmax(scores))]
    mask_small = (best.astype(np.uint8) * 255)
    mask_full = cv2.resize(mask_small, original.size, interpolation=cv2.INTER_NEAREST)
    print(f"[sam] points={len(points)} best_score={float(scores.max()):.3f} mask_px={int(mask_full.sum()/255)}")
    return Image.fromarray(mask_full, mode="L")


def run_sam_on_box(original_img, original: Image.Image, box_xyxy: tuple[int, int, int, int]) -> Image.Image:
    """用 SAM box prompt 分割矩形内目标。"""
    predictor = get_sam_predictor()
    scale = _ensure_sam_embedding(original_img, original)
    x1, y1, x2, y2 = box_xyxy
    box = np.array([x1 * scale, y1 * scale, x2 * scale, y2 * scale], dtype=np.float32)

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )
    best = masks[int(np.argmax(scores))]
    mask_small = (best.astype(np.uint8) * 255)
    mask_full = cv2.resize(mask_small, original.size, interpolation=cv2.INTER_NEAREST)
    print(f"[sam] box={box_xyxy} best_score={float(scores.max()):.3f} mask_px={int(mask_full.sum()/255)}")
    return Image.fromarray(mask_full, mode="L")


def _union_masks(a: Image.Image | None, b: Image.Image | None, size: tuple[int, int]) -> Image.Image | None:
    if a is None and b is None:
        return None
    w, h = size
    base = np.zeros((h, w), dtype=np.uint8)
    for m in (a, b):
        if m is None:
            continue
        arr = np.array(m.convert("L").resize((w, h), Image.Resampling.NEAREST))
        base = np.maximum(base, arr)
    return Image.fromarray(base, mode="L")


def _compose_editor(original: Image.Image, mask: Image.Image | None):
    if mask is None:
        preview = _preview_size(original).convert("RGBA")
        return {"background": preview, "layers": [], "composite": preview}
    return mask_to_editor(original, mask)


def on_sam_click(
    original_img,
    points_state,
    point_mode,
    rects_state,
    rect_pending,
    session_id,
    evt: gr.SelectData,
):
    """在原图预览上点击，追加点并立即跑 SAM。Mask 留在服务端。"""
    original = _to_pil_rgb(original_img)
    if original is None:
        raise gr.Error("请先上传全景原图")

    sid = _ensure_session_id(session_id)
    ox, oy = _preview_to_original_xy(original, evt.index)
    label = 1 if point_mode == "正点（人像上）" else 0

    points = list(points_state or [])
    points.append((ox, oy, label))
    rects = list(rects_state or [])
    _, rect_mask = _get_masks(sid)

    points_mask = run_sam_on_points(original_img, original, points)
    mask = _union_masks(points_mask, rect_mask, original.size)
    _put_masks(sid, mask, rect_mask)
    vis = overlay_mask_preview(original, mask, points, rects, rect_pending)
    status = (
        f"SAM 已更新：{len(points)} 个点（最近 {'正' if label == 1 else '负'}点 @ ({ox},{oy})）。"
        "可继续点击。笔刷画板不会每次回传，精修前请点「载入当前 Mask 到笔刷」。"
    )
    return points, rects, rect_pending, vis, status, sid


def on_rect_click(
    original_img,
    points_state,
    rects_state,
    rect_pending,
    session_id,
    rect_use_sam,
    evt: gr.SelectData,
):
    """对角两点定矩形框：第 1 点定点，第 2 点成框并写入 Mask。"""
    original = _to_pil_rgb(original_img)
    if original is None:
        raise gr.Error("请先上传全景原图")

    sid = _ensure_session_id(session_id)
    ox, oy = _preview_to_original_xy(original, evt.index)
    points = list(points_state or [])
    rects = list(rects_state or [])
    pending = rect_pending
    combined, rect_mask = _get_masks(sid)

    if pending is None:
        pending = (ox, oy)
        vis = overlay_mask_preview(original, combined, points, rects, pending)
        status = f"矩形起点已定 @ ({ox},{oy})，请再点对角终点完成框选。"
        return points, rects, pending, vis, status, sid

    x1, y1, x2, y2 = _normalize_rect(pending[0], pending[1], ox, oy)
    if x2 - x1 < 4 or y2 - y1 < 4:
        vis = overlay_mask_preview(original, combined, points, rects, pending)
        status = "矩形太小，请重新点击更大的对角区域。"
        return points, rects, pending, vis, status, sid

    rect = (x1, y1, x2, y2)
    rects.append(rect)
    pending = None

    if rect_use_sam:
        box_mask = run_sam_on_box(original_img, original, rect)
        rect_mask = _union_masks(rect_mask, box_mask, original.size)
        how = "SAM 框内分割"
    else:
        rect_mask = union_rect_mask(rect_mask, original.size, rect)
        how = "整框填色"

    points_mask = run_sam_on_points(original_img, original, points) if points else None
    mask = _union_masks(points_mask, rect_mask, original.size)
    _put_masks(sid, mask, rect_mask)
    vis = overlay_mask_preview(original, mask, points, rects, pending)
    status = f"已添加矩形 #{len(rects)} [{how}] ({x1},{y1})-({x2},{y2})。可继续画框。精修前请载入 Mask 到笔刷。"
    return points, rects, pending, vis, status, sid


def on_interact_click(
    original_img,
    points_state,
    point_mode,
    interact_mode,
    rects_state,
    rect_pending,
    session_id,
    rect_use_sam,
    evt: gr.SelectData,
):
    if interact_mode == "矩形框":
        return on_rect_click(
            original_img,
            points_state,
            rects_state,
            rect_pending,
            session_id,
            rect_use_sam,
            evt,
        )
    return on_sam_click(
        original_img, points_state, point_mode, rects_state, rect_pending, session_id, evt
    )


def sync_brush_editor(original_img, session_id):
    """仅在用户要笔刷精修时把 Mask 写入画板，避免每次点选回传 ImageEditor。"""
    original = _to_pil_rgb(original_img)
    if original is None:
        raise gr.Error("请先上传全景原图")
    mask, _ = _get_masks(session_id)
    return _compose_editor(original, mask)


def clear_sam(original_img, session_id):
    global _SAM_EMBED_KEY
    _SAM_EMBED_KEY = None
    sid = _ensure_session_id(session_id)
    _reset_masks(sid)
    original = _to_pil_rgb(original_img)
    points: list = []
    rects: list = []
    pending = None
    if original is None:
        return points, rects, pending, None, None, "已清空", sid
    preview = _preview_size(original)
    editor = {
        "background": preview.convert("RGBA"),
        "layers": [],
        "composite": preview.convert("RGBA"),
    }
    return points, rects, pending, preview, editor, "已清空 SAM 点、矩形框与 Mask，可重新标注", sid


def _model_controls(model: str):
    if model == "PowerPaint":
        return (
            gr.update(
                minimum=512,
                maximum=1280,
                value=POWERPAINT_MAX_SIDE,
                step=128,
                label="PowerPaint ROI 最长边",
                info="本机 16GB 建议 768；更大更清晰也更吃内存",
            ),
            "PowerPaint：按 Mask 裁 ROI 后推理。首次加载约 1～2 分钟，之后每次约 20～60 秒。需已安装旁路 iopaint-bench。",
        )
    if model == "Qwen-Edit":
        return (
            gr.update(
                minimum=512,
                maximum=1024,
                value=QWEN_EDIT_MAX_SIDE,
                step=128,
                label="Qwen-Edit ROI 最长边",
                info="3090 建议 768；禁止送整幅全景。与 PowerPaint 互斥占显存。",
            ),
            "Qwen-Edit：按 Mask 裁 ROI 后用指令编辑补背景。首次加载量化权重较慢；默认不随服务预热。",
        )
    return (
        gr.update(
            minimum=1024,
            maximum=2560,
            value=MAX_INFER_SIDE,
            step=256,
            label="推理最长边",
            info="Mac 建议 1536；内存充足可试 2048",
        ),
        "LaMa：整图降采样推理，速度快，适合日常。换模型不必重新点选。",
    )


def remove_person(
    original_img,
    editor_value,
    session_id,
    dilate_px: int,
    max_side: int,
    model_name: str,
    source_stem: str,
    progress=gr.Progress(),
):
    clock = StageClock()
    reset_torch_peak()
    progress(0.05, desc="读取原图...")
    original = _to_pil_rgb(original_img)
    if original is None:
        raise gr.Error("请先上传全景原图（左侧「原图」）")
    clock.tick("read")

    progress(0.15, desc="解析 Mask...")
    brush_mask = _extract_brush_mask(editor_value, original.size)
    sam_mask, _ = _get_masks(session_id)
    if sam_mask is not None:
        sam_mask = sam_mask.convert("L").resize(original.size, Image.Resampling.NEAREST)

    if brush_mask is not None and sam_mask is not None:
        # 画板精修优先：有笔刷层则用笔刷（已含 SAM/矩形写入的红层）
        mask = brush_mask
    elif brush_mask is not None:
        mask = brush_mask
    elif sam_mask is not None and np.array(sam_mask).max() > 0:
        mask = sam_mask
    else:
        raise gr.Error("请先用 SAM 点选 / 矩形框标注，或在画板笔刷涂抹")

    mask = dilate_mask(mask, int(dilate_px))
    mask_bin = mask.point(lambda p: 255 if p > 127 else 0)
    clock.tick("mask")

    backend = get_backend(model_name)
    progress(0.3, desc=f"{backend.name} 准备中（最长边≤{int(max_side)}）...")
    print(f"[inpaint] model={backend.name} original={original.size} max_side={int(max_side)}")

    progress(0.45, desc=f"{backend.name} 消除中（进度会停在这直到推理结束）...")
    try:
        result = backend.inpaint(original, mask_bin, int(max_side))
    except FileNotFoundError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:
        raise gr.Error(f"{backend.name} 失败：{e}") from e
    clock.tick("inpaint")

    progress(0.85, desc="保存结果...")
    stem = source_stem if source_stem and source_stem != "upload" else _image_stem(original_img)
    out_path, mask_path = _output_paths(stem, backend.name)
    result.save(out_path, quality=95)
    mask_bin.save(mask_path)
    jpeg_mb = out_path.stat().st_size / (1024 * 1024)
    clock.tick("save")

    progress(0.95, desc="准备预览...")
    mask_prev = _preview_size(mask_bin.convert("RGB"))
    clock.tick("preview")
    res = format_resources()
    print(
        f"[profile] {backend.name} {clock.summary()} | "
        f"jpeg={jpeg_mb:.1f}MB {result.size[0]}x{result.size[1]} | {res}"
    )
    print(
        f"[inpaint] saved {out_path.name} mask={mask_path.name} "
        f"size={result.size} model={backend.name}"
    )

    progress(1.0, desc="完成")
    status = (
        f"{backend.name} 完成 {result.size[0]}×{result.size[1]} | "
        f"{clock.summary()} | jpeg={jpeg_mb:.1f}MB | 已保存 {out_path.name}"
    )
    return str(out_path), mask_prev, status


def load_original(img, session_id):
    """上传原图后：刷新预览与空画板。"""
    global _SAM_EMBED_KEY
    _SAM_EMBED_KEY = None
    sid = _ensure_session_id(session_id)
    _reset_masks(sid)
    pil = _to_pil_rgb(img)
    stem = _image_stem(img)
    if pil is None:
        return [], [], None, None, None, "请上传原图", "upload", sid
    preview = _preview_size(pil)
    editor = {
        "background": preview.convert("RGBA"),
        "layers": [],
        "composite": preview.convert("RGBA"),
    }
    return (
        [],
        [],
        None,
        preview,
        editor,
        f"已加载 {pil.size[0]}×{pil.size[1]}（{stem}）。可选「SAM 点选」或「矩形框」标注人像。",
        stem,
        sid,
    )


def on_mode_change(interact_mode):
    if interact_mode == "矩形框":
        tip = "【矩形框】请在下方蓝色标题的「标注区」上操作：先点一个角，再点对角 → 出现青色框。不要用笔刷画板（那是涂红点用的）。"
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(label="标注区 · 矩形框：点角1 → 再点对角2（可多框）"),
            tip,
        )
    tip = "【SAM 点选】请在下方「标注区」人像上单击（可多点）。笔刷画板仅用于事后精修。"
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(label="标注区 · SAM 点选：在人像上单击"),
        tip,
    )


CUSTOM_CSS = """
.gradio-container { max-width: 1500px !important; }
footer { display: none !important; }
#interact-view label span { color: #0b5fff !important; font-weight: 700 !important; }
"""

with gr.Blocks(title="全景人像消除 · SAM + 可切换修复模型") as demo:
    gr.Markdown(
        """
        # 全景人像消除（SAM 点选 / 矩形框 + 笔刷精修 + 可切换修复模型）

        1. 上传全景原图  
        2. 选标注模式后，在中间 **标注区**（蓝色标题那张图）上操作——**不是**下面带笔刷工具栏的画板  
           - **SAM 点选**：单击人像  
           - **矩形框**：点两个**对角点**定框（会出现青色矩形）  
        3. 需要修边时再展开「笔刷精修」，点「载入当前 Mask 到笔刷」  
        4. 选修复模型后开始消除 → `output/原图名_模型_时间.jpg`  
           Mask 留在服务端，点选只回传 960 预览。
        """
    )

    points_state = gr.State([])
    rects_state = gr.State([])
    rect_pending_state = gr.State(None)
    session_id = gr.State("")
    source_stem_state = gr.State("upload")

    with gr.Row():
        with gr.Column(scale=1):
            original = gr.Image(
                label="原图（保持分辨率）",
                type="filepath",
                image_mode="RGB",
                height=220,
            )
            interact_mode = gr.Radio(
                ["SAM 点选", "矩形框"],
                value="SAM 点选",
                label="标注模式",
            )
            point_mode = gr.Radio(
                ["正点（人像上）", "负点（排除背景）"],
                value="正点（人像上）",
                label="SAM 点击模式",
            )
            rect_use_sam = gr.Checkbox(
                value=False,
                label="矩形框内用 SAM 分割（更贴合人像；关闭则整框消除）",
                visible=False,
            )
            clear_btn = gr.Button("清空标注 / Mask")
            model_name = gr.Radio(
                choices=list_backend_names(),
                value=DEFAULT_MODEL if DEFAULT_MODEL in list_backend_names() else "LaMa",
                label="修复模型",
            )
            model_tip = gr.Markdown("LaMa：整图降采样推理，速度快，适合日常。换模型不必重新点选。")
            dilate_px = gr.Slider(0, 30, value=6, step=1, label="Mask 膨胀（像素）")
            max_side = gr.Slider(
                1024,
                2560,
                value=MAX_INFER_SIDE,
                step=256,
                label="推理最长边",
                info="Mac 建议 1536；内存充足可试 2048",
            )
            run_btn = gr.Button("开始消除", variant="primary")
            status = gr.Textbox(label="状态", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### 在此图上标注（点选 / 画框）")
            sam_view = gr.Image(
                label="标注区 · SAM 点选：在人像上单击",
                type="pil",
                format="jpeg",
                height=480,
                elem_id="interact-view",
            )
            with gr.Accordion("可选：笔刷精修 Mask（点选时不回传画板）", open=False):
                load_brush_btn = gr.Button("载入当前 Mask 到笔刷")
                editor = gr.ImageEditor(
                    label="笔刷精修（仅涂抹/橡皮；不能画矩形框）",
                    type="pil",
                    image_mode="RGBA",
                    brush=gr.Brush(default_size=28, colors=["#ff0000"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=28),
                    layers=True,
                    fixed_canvas=False,
                    height=280,
                )

        with gr.Column(scale=1):
            result = gr.Image(label="消除结果（原分辨率 JPEG）", type="filepath", height=420)
            mask_preview = gr.Image(label="最终 Mask 预览", type="pil", height=200)

    _examples = list_resource_examples()
    if _examples:
        gr.Examples(
            examples=_examples,
            inputs=[original],
            label=f"样例全景（resource/，共 {len(_examples)} 张）",
        )

    interact_outputs = [
        points_state,
        rects_state,
        rect_pending_state,
        sam_view,
        editor,
        status,
    ]
    click_outputs = [
        points_state,
        rects_state,
        rect_pending_state,
        sam_view,
        status,
        session_id,
    ]

    original.change(
        fn=load_original,
        inputs=[original, session_id],
        outputs=interact_outputs + [source_stem_state, session_id],
    )
    interact_mode.change(
        fn=on_mode_change,
        inputs=[interact_mode],
        outputs=[point_mode, rect_use_sam, sam_view, status],
    )
    model_name.change(
        fn=_model_controls,
        inputs=[model_name],
        outputs=[max_side, model_tip],
    )
    sam_view.select(
        fn=on_interact_click,
        inputs=[
            original,
            points_state,
            point_mode,
            interact_mode,
            rects_state,
            rect_pending_state,
            session_id,
            rect_use_sam,
        ],
        outputs=click_outputs,
    )
    load_brush_btn.click(
        fn=sync_brush_editor,
        inputs=[original, session_id],
        outputs=[editor],
    )
    clear_btn.click(
        fn=clear_sam,
        inputs=[original, session_id],
        outputs=interact_outputs + [session_id],
    )
    run_btn.click(
        fn=remove_person,
        inputs=[original, editor, session_id, dilate_px, max_side, model_name, source_stem_state],
        outputs=[result, mask_preview, status],
    )


if __name__ == "__main__":
    def warmup_models() -> None:
        print("[warmup] SAM ...")
        try:
            get_sam_predictor()
        except Exception as e:
            print(f"[warmup] SAM 失败: {e}")
        names = list_backend_names()
        both_heavy = "PowerPaint" in names and "Qwen-Edit" in names
        for name in names:
            if name == "Qwen-Edit" and os.environ.get("WARMUP_QWEN", "0") != "1":
                print("[warmup] Qwen-Edit 跳过（WARMUP_QWEN!=1，首次使用时再加载）")
                continue
            if (
                name == "PowerPaint"
                and both_heavy
                and os.environ.get("WARMUP_POWERPAINT", "0") != "1"
            ):
                print("[warmup] PowerPaint 跳过（与 Qwen 共存时不预热，首次使用再加载；点 Qwen 也会卸掉 PowerPaint）")
                continue
            print(f"[warmup] {name} ...")
            backend = get_backend(name)
            warmup = getattr(backend, "warmup", None)
            if not callable(warmup):
                continue
            try:
                warmup()
            except Exception as e:
                print(f"[warmup] {name} 失败（服务仍启动，首次使用该模型时会再试）: {e}")
        print("[warmup] 完成，开始接受请求")

    if os.environ.get("WARMUP", "1") != "0":
        warmup_models()
    host = os.environ.get("HOST", "127.0.0.1")
    inbrowser = os.environ.get("INBROWSER", "1" if host in {"127.0.0.1", "localhost"} else "0") == "1"
    auth = None
    auth_raw = os.environ.get("GRADIO_AUTH", "").strip()
    if auth_raw and ":" in auth_raw:
        user, password = auth_raw.split(":", 1)
        auth = (user, password)
    demo.queue(max_size=2).launch(
        server_name=host,
        server_port=int(os.environ.get("PORT", "7860")),
        inbrowser=inbrowser,
        auth=auth,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
