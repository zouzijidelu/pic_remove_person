"""全景人像消除 Demo：人工笔刷 Mask + LaMa（保持原图分辨率）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 推理最长边上限（过大易被 macOS 因内存杀掉；1536 通常够用且远高于 Hama 720）
MAX_INFER_SIDE = int(os.environ.get("MAX_INFER_SIDE", "1536"))


def pick_device() -> torch.device:
    # TorchScript LaMa 在 MPS 上常出块状/花屏伪影，默认强制 CPU 保证效果
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
    print(f"[lama-demo] loading LaMa on {device} ...")
    return SimpleLama(device=device)


def _to_pil_rgb(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    return Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")


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
            # 只认 alpha：避免把整层不透明底当成 mask
            alpha = np.array(layer_img.split()[-1])
            if alpha.max() == 0:
                # 部分 Gradio 版本笔刷写在 RGB、alpha 全 255 或全 0
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


def resize_for_infer(image: Image.Image, mask: Image.Image, max_side: int):
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_side:
        return image, mask, 1.0
    scale = max_side / float(long_side)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    # 对齐到 8，减少 pad 浪费
    nw = max(8, nw - nw % 8)
    nh = max(8, nh - nh % 8)
    image_s = image.resize((nw, nh), Image.Resampling.LANCZOS)
    mask_s = mask.resize((nw, nh), Image.Resampling.NEAREST)
    return image_s, mask_s, scale


def run_lama(image: Image.Image, mask: Image.Image) -> Image.Image:
    """在给定分辨率上跑 LaMa，并裁掉 pad，保证输出尺寸=输入。"""
    from simple_lama_inpainting.utils.util import prepare_img_and_mask

    lama = get_lama()
    orig_w, orig_h = image.size
    img_t, mask_t = prepare_img_and_mask(image, mask, lama.device, pad_out_to_modulo=8)

    with torch.inference_mode():
        inpainted = lama.model(img_t, mask_t)
        out = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)

    # 去掉 pad
    out = out[:orig_h, :orig_w]
    return Image.fromarray(out, mode="RGB")


def paste_back(original: Image.Image, inpainted_small: Image.Image, mask_full: Image.Image) -> Image.Image:
    """若推理用了降采样，把结果放大后仅在 mask 区域贴回原图，避免整图变糊。"""
    if inpainted_small.size == original.size:
        return inpainted_small

    up = inpainted_small.resize(original.size, Image.Resampling.LANCZOS)
    orig = np.array(original.convert("RGB"))
    up_arr = np.array(up.convert("RGB"))
    m = np.array(mask_full.resize(original.size, Image.Resampling.NEAREST))
    # 软边缘混合，减轻贴回接缝
    m_f = (m.astype(np.float32) / 255.0)[..., None]
    # 轻微羽化
    if m_f.max() > 0:
        m_blur = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=2)
        m_f = (m_blur / 255.0)[..., None]
    blended = orig.astype(np.float32) * (1.0 - m_f) + up_arr.astype(np.float32) * m_f
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def remove_person(
    original_img,
    editor_value,
    dilate_px: int,
    max_side: int,
    progress=gr.Progress(),
):
    progress(0.05, desc="读取原图...")
    original = _to_pil_rgb(original_img)
    if original is None:
        raise gr.Error("请先上传全景原图（左侧「原图」）")

    progress(0.15, desc="解析笔刷 Mask...")
    mask = _extract_brush_mask(editor_value, original.size)
    if mask is None:
        raise gr.Error("请在中间画板用笔刷涂抹要消除的人像（含脚下阴影更佳）")

    mask = dilate_mask(mask, int(dilate_px))
    mask_bin = mask.point(lambda p: 255 if p > 127 else 0)

    progress(0.3, desc=f"准备推理（最长边≤{int(max_side)}）...")
    infer_img, infer_mask, scale = resize_for_infer(original, mask_bin, int(max_side))
    print(f"[lama-demo] original={original.size} infer={infer_img.size} scale={scale:.3f} device={pick_device()}")

    progress(0.45, desc="LaMa 消除中...")
    inpainted = run_lama(infer_img, infer_mask)

    progress(0.85, desc="贴回原图分辨率...")
    result = paste_back(original, inpainted, mask_bin)

    out_path = OUTPUT_DIR / "last_result.jpg"
    result.save(out_path, quality=95)
    mask_bin.save(OUTPUT_DIR / "last_mask.png")
    print(f"[lama-demo] saved {out_path} size={result.size}")

    progress(1.0, desc="完成")
    # mask 预览缩略
    mask_preview = mask_bin.convert("RGB")
    return result, mask_preview, f"输出 {result.size[0]}×{result.size[1]}，已保存到 output/last_result.jpg"


def load_to_editor(img):
    """上传原图后，同步到画板供涂抹（画板仅作标注，不作为最终分辨率来源）。"""
    pil = _to_pil_rgb(img)
    if pil is None:
        return None
    # 画板用稍小预览即可，加快涂抹；真正推理用左侧原图
    w, h = pil.size
    max_preview = 1600
    if max(w, h) > max_preview:
        s = max_preview / max(w, h)
        pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)
    return {
        "background": pil.convert("RGBA"),
        "layers": [],
        "composite": pil.convert("RGBA"),
    }


CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="全景人像消除 · LaMa") as demo:
    gr.Markdown(
        """
        # 全景人像消除（人工 Mask + LaMa）

        **重要**：最终分辨率以左侧「原图」为准，不再被画板压缩（避免马赛克）。

        1. 上传全景原图  
        2. 在画板用笔刷涂抹人像（含阴影）  
        3. 点击开始消除 → 结果保存到 `output/last_result.jpg`

        设备默认 **CPU**（避免 Apple MPS 块状花屏）。可用环境变量 `LAMA_DEVICE=mps` 尝试加速。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            original = gr.Image(
                label="原图（保持分辨率）",
                type="pil",
                image_mode="RGB",
                height=280,
            )
            dilate_px = gr.Slider(0, 30, value=6, step=1, label="Mask 膨胀（像素）")
            max_side = gr.Slider(
                1024,
                2560,
                value=MAX_INFER_SIDE,
                step=256,
                label="推理最长边",
                info="越大越清晰，但越占内存。Mac 建议 1536；内存充足可试 2048",
            )
            run_btn = gr.Button("开始消除", variant="primary")
            status = gr.Textbox(label="状态", interactive=False)

        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                label="在此涂抹人像（仅用于画 Mask）",
                type="pil",
                image_mode="RGBA",
                brush=gr.Brush(default_size=28, colors=["#ff0000"], color_mode="fixed"),
                eraser=gr.Eraser(default_size=28),
                layers=True,
                fixed_canvas=False,
                height=420,
            )

        with gr.Column(scale=1):
            result = gr.Image(label="消除结果（原图像素）", type="pil", height=420)
            mask_preview = gr.Image(label="Mask 预览", type="pil", height=200)

    gr.Examples(
        examples=[
            ["resource/44290365location_07.jpg"],
            ["resource/505018258location_10.jpg"],
            ["resource/826163191location_04.jpg"],
        ],
        inputs=[original],
        label="样例全景",
    )

    original.change(fn=load_to_editor, inputs=[original], outputs=[editor])
    run_btn.click(
        fn=remove_person,
        inputs=[original, editor, dilate_px, max_side],
        outputs=[result, mask_preview, status],
    )


if __name__ == "__main__":
    demo.queue(max_size=2).launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PORT", "7860")),
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
