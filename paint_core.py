"""Offline non-destructive photo operations for the manual manga editor.

Selections never trigger OCR and never erase a complete rectangle. Pixel edits
are always applied to a working copy; the source file is never overwritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

Box = tuple[int, int, int, int]


def clip_box(box: Box, size: tuple[int, int]) -> Box:
    x, y, w, h = (int(v) for v in box)
    width, height = size
    x1 = min(width, max(0, x))
    y1 = min(height, max(0, y))
    x2 = min(width, max(x1, x + max(0, w)))
    y2 = min(height, max(y1, y + max(0, h)))
    return x1, y1, x2 - x1, y2 - y1


def ink_mask(image: Image.Image, box: Box) -> tuple[Box, np.ndarray]:
    """Conservative per-character mask within a user-picked rectangle.

    Drops components touching the selection margin (usually balloon borders),
    large regions (illustration/balloon), and tiny noise. No OCR assumptions.
    A user can always restore via the original-image brush.
    """
    x, y, w, h = clip_box(box, image.size)
    mask = np.zeros((h, w), dtype=np.uint8)
    if w < 9 or h < 9:
        return (x, y, w, h), mask
    rgb = np.asarray(image.crop((x, y, x + w, y + h)).convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sigma = max(3, min(w, h) / 9.0)
    local_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    delta = local_bg.astype(np.int16) - gray.astype(np.int16)
    margin = max(2, min(w, h) // 50)
    area = w * h

    def components(binary):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        chosen = np.zeros(binary.shape, dtype=np.uint8)
        for i in range(1, count):
            cx, cy, cw, ch, a = (int(v) for v in stats[i])
            if a < 2 or a > area * .10:
                continue
            if cx <= margin or cy <= margin or cx + cw >= w - margin or cy + ch >= h - margin:
                continue
            if cw > w * .44 or ch > h * .72:
                continue
            if a / max(1, cw * ch) > .95 and cw > w * .22:
                continue
            chosen[labels == i] = 255
        return chosen

    # Negative contrast is common manga ink. Positive contrast handles light
    # lettering on dark panels; choose only one polarity to avoid destroying
    # both text and background details in the same selection.
    dark = components(np.uint8(delta >= 15) * 255)
    light = components(np.uint8(delta <= -18) * 255)
    dn = cv2.countNonZero(dark)
    ln = cv2.countNonZero(light)
    if dn == 0 and ln == 0:
        return (x, y, w, h), mask
    if dn >= ln and dn:
        # Manga glyphs are often black with a thin WHITE halo. Remove the halo
        # only when it is next to a detected dark stroke; never indiscriminately
        # remove every bright area in the balloon.
        neighbours = cv2.dilate(dark, np.ones((9, 9), np.uint8))
        mask = cv2.bitwise_or(dark, cv2.bitwise_and(light, neighbours))
    else:
        mask = light
    # Include anti-alias pixels and outlined text edges conservatively.
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=2)
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    return (x, y, w, h), mask


def clear_letters(image: Image.Image, box: Box) -> tuple[Image.Image, int, Image.Image]:
    """Inpaint detected glyph pixels only, return (new, pixel count, mask)."""
    rect, mask = ink_mask(image, box)
    x, y, w, h = rect
    if not w or not h or not np.any(mask):
        return image.copy(), 0, Image.fromarray(mask, 'L')
    rgba = image.convert('RGBA')
    src = np.array(rgba.crop((x, y, x + w, y + h)))
    repaired = cv2.inpaint(src[:, :, :3], mask, 4, cv2.INPAINT_TELEA)
    src[mask > 0, :3] = repaired[mask > 0, :3]
    output = rgba.copy()
    output.paste(Image.fromarray(src, 'RGBA'), (x, y))
    if image.mode == 'RGB':
        output = output.convert('RGB')
    return output, int(np.count_nonzero(mask)), Image.fromarray(mask, 'L')


def brush_segment(image: Image.Image, original: Image.Image, start, end,
                  diameter: int, color: tuple[int, int, int, int], restore=False) -> Image.Image:
    """Paint or restore an antialiased round stroke without changing source."""
    result = image.convert('RGBA').copy()
    radius = max(1, int(diameter) // 2)
    x0, y0 = (round(v) for v in start)
    x1, y1 = (round(v) for v in end)
    if restore:
        mask = Image.new('L', result.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.line((x0, y0, x1, y1), fill=255, width=radius * 2)
        for px, py in ((x0, y0), (x1, y1)):
            draw.ellipse((px-radius,py-radius,px+radius,py+radius), fill=255)
        result.paste(original.convert('RGBA'), (0, 0), mask)
    else:
        draw = ImageDraw.Draw(result)
        draw.line((x0, y0, x1, y1), fill=color, width=radius * 2)
        for px, py in ((x0, y0), (x1, y1)):
            draw.ellipse((px-radius,py-radius,px+radius,py+radius), fill=color)
    return result.convert('RGB') if image.mode == 'RGB' else result


def palette_smudge(image: Image.Image, start, end, diameter: int) -> Image.Image:
    """Local palette-knife drag: smear colors sampled just behind brush movement."""
    radius = max(2, min(100, diameter // 2))
    x1, y1 = (round(v) for v in end)
    dx = int(round(end[0] - start[0]))
    dy = int(round(end[1] - start[1]))
    if not dx and not dy:
        return image.copy()
    src = np.asarray(image.convert('RGBA')).copy()
    h, w = src.shape[:2]
    lx, ly = max(0, x1 - radius), max(0, y1 - radius)
    rx, ry = min(w, x1 + radius + 1), min(h, y1 + radius + 1)
    if lx >= rx or ly >= ry:
        return image.copy()
    yy, xx = np.mgrid[ly:ry, lx:rx]
    radial = np.maximum(0.0, 1.0 - ((xx - x1) ** 2 + (yy - y1) ** 2) / radius ** 2)
    alpha = (radial ** .6 * .8)[..., None]
    fromx = np.clip(xx - dx, 0, w - 1)
    fromy = np.clip(yy - dy, 0, h - 1)
    patch = src[ly:ry, lx:rx].astype(np.float32)
    smear = src[fromy, fromx].astype(np.float32)
    src[ly:ry, lx:rx] = np.uint8(np.clip(patch * (1 - alpha) + smear * alpha, 0, 255))
    out = Image.fromarray(src, 'RGBA')
    return out.convert('RGB') if image.mode == 'RGB' else out