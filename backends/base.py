"""修复后端协议。SAM / 笔刷 / 贴回原图与具体模型无关。"""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class InpaintBackend(Protocol):
    name: str

    def inpaint(self, image: Image.Image, mask: Image.Image, max_side: int) -> Image.Image:
        """image/mask 为原图尺寸；白=消除。返回与原图同尺寸的 RGB。"""
        ...
