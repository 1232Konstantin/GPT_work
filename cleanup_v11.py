"""v0.11: independent ink-mask discovery and background reconstruction.

The two stages never change pixels outside the explicitly approved mask.
Search uses either background samples on the selection border or a robust
estimate across its interior. Reconstruction uses the CURRENT pixel image.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from contrast_cleanup import Segmentation, repair_background


def _bounds(image: Image.Image, region):
    x, y, w, h = map(int, region)
    return (max(0, x), max(0, y),
            min(image.width, x + w), min(image.height, y + h))


def _boundary_samples(rgb):
    if min(rgb.shape[:2]) < 2:
        return rgb.reshape(-1, 3)
    return np.concatenate((rgb[0], rgb[-1], rgb[1:-1, 0],
                           rgb[1:-1, -1]), axis=0)


def _components(binary, *, max_fraction=0.98):
    """Keep actual foreground components, even when selection clips glyphs.

    A user's tight rectangle can cut a character or a coloured stroke at
    any edge. Touching that edge is NOT evidence that the stroke is noise.
    """
    height, width = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(binary), connectivity=8)
    accepted = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 2 <= area <= width * height * max_fraction:
            accepted[labels == i] = 255
    return accepted


def _fill_outlined_interiors(mask):
    """Include enclosed background regions, without filling open shapes.

    A 3-pixel closing seals tiny holes in anti-aliased outlines. The holes
    themselves are identified by connected background components which do
    not reach the selection edge. We never fill an arbitrary external contour.
    """
    h, w = mask.shape
    if h < 3 or w < 3:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    inverse = np.uint8(closed == 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=8)
    filled = mask.copy()
    for i in range(1, n):
        x, y, cw, ch, area = map(int, stats[i])
        if area < 4 or area > w * h * .28:
            continue
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue
        if cw > w * .85 or ch > h * .85:
            continue
        filled[labels == i] = 255
        # Seal only the tiny breaks adjacent to an accepted hole.
        vicinity = cv2.dilate(np.uint8(labels == i),
                              kernel, iterations=1) > 0
        filled[vicinity & (closed > 0)] = 255
    return filled


def segment_text(image: Image.Image, region, *,
                 reference: str = 'area', outlined: bool = False,
                 grow: int = 1) -> Segmentation:
    """Find dark/light and coloured strokes against border or interior estimate."""
    if reference not in ('border', 'area'):
        raise ValueError('Unknown background reference')
    rgb = np.asarray(image.convert('RGB'))
    x0, y0, x1, y1 = _bounds(image, region)
    empty = np.zeros(rgb.shape[:2], dtype=np.uint8)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return Segmentation(empty, empty.copy(), empty.copy(), rgb.copy())
    crop = rgb[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    if reference == 'border':
        samples = np.concatenate((lab[0], lab[-1],
                                  lab[1:-1, 0], lab[1:-1, -1]), axis=0)
    else:
        samples = lab.reshape(-1, 3)
    # Robust background estimate: border samples for tight text crops,
    # interior samples for a background-dominated user selection.
    base = np.median(samples, axis=0)
    luminance_delta = lab[:, :, 0] - base[0]
    chroma_delta = np.linalg.norm(lab[:, :, 1:3] - base[1:3], axis=2)

    # Set thresholds by background variation, rather than fixed values
    # demanding strongly contrasting lettering on every page. Floors avoid
    # classifying tiny RGB/printing noise as foreground. Caps allow lettering
    # to remain detectable on moderately varying backgrounds.
    luminance_spread = float(np.median(np.abs(samples[:, 0] - base[0])))
    colour_spread = float(np.median(np.linalg.norm(
        samples[:, 1:3] - base[1:3], axis=1)))
    luminance_threshold = float(np.clip(10.0 + 2.0 * luminance_spread,
                                         12.0, 34.0))
    colour_threshold = float(np.clip(12.0 + 2.0 * colour_spread,
                                      14.0, 35.0))
    dark_pixels = luminance_delta <= -luminance_threshold
    light_pixels = luminance_delta >= luminance_threshold
    different_colour = chroma_delta >= colour_threshold

    # Three INDEPENDENT reasons to mark a pixel: dark ink, light ink, or
    # different hue. Never require a second luminance/local-edge condition
    # for colour: it would erase the solid middle of pink-on-gray glyphs.
    candidates = dark_pixels | light_pixels | different_colour
    core = _components(candidates)
    if outlined:
        core = _fill_outlined_interiors(core)
    if grow > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        core = cv2.dilate(core, kernel)
    full = empty.copy()
    full[y0:y1, x0:x1] = core
    # Output contracts remain compatible with the earlier segmentation API.
    dark = empty.copy()
    light = empty.copy()
    dark[y0:y1, x0:x1] = np.uint8((core > 0) & dark_pixels) * 255
    light[y0:y1, x0:x1] = np.uint8((core > 0) & light_pixels) * 255
    return Segmentation(full, dark, light, rgb.copy())


def _reference_colour(rgb, mask, region, *, grid=False):
    """Exclude ALL removed pixels from background colour measurements."""
    x0, y0, x1, y1 = region
    roi = rgb[y0:y1, x0:x1]
    protected = ~mask[y0:y1, x0:x1]
    if grid:
        yy = np.unique(np.linspace(0, max(0, roi.shape[0] - 1),
                                   min(18, roi.shape[0]), dtype=int))
        xx = np.unique(np.linspace(0, max(0, roi.shape[1] - 1),
                                   min(18, roi.shape[1]), dtype=int))
        points = roi[np.ix_(yy, xx)]
        ok = protected[np.ix_(yy, xx)]
        samples = points[ok]
        if len(samples) < 5:
            samples = roi[protected]
    else:
        edge = np.zeros(protected.shape, dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        samples = roi[edge & protected]
        if len(samples) < 5:
            samples = roi[protected]
    if samples.size == 0:
        return None, None
    return np.mean(samples.astype(np.float32), axis=0), samples.astype(np.float32)


def _fill_axis(rgb, mask, region, *, horizontal):
    """Interpolate each masked run from the nearest untouched pixels."""
    out = rgb.copy().astype(np.float32)
    x0, y0, x1, y1 = region
    baseline, _ = _reference_colour(rgb, mask, region)
    if baseline is None:
        baseline = np.mean(rgb.reshape(-1, 3), axis=0)
    if horizontal:
        for y in range(y0, y1):
            x = x0
            while x < x1:
                if not mask[y, x]:
                    x += 1
                    continue
                start = x
                while x < x1 and mask[y, x]:
                    x += 1
                end = x
                left = rgb[y, start - 1] if start > 0 and not mask[y, start - 1] else None
                right = rgb[y, end] if end < rgb.shape[1] and not mask[y, end] else None
                if left is None and right is None:
                    left = right = baseline
                elif left is None:
                    left = right
                elif right is None:
                    right = left
                for column in range(start, end):
                    t = (column - start + 1) / (end - start + 1)
                    out[y, column] = np.asarray(left) * (1 - t) + np.asarray(right) * t
    else:
        for x in range(x0, x1):
            y = y0
            while y < y1:
                if not mask[y, x]:
                    y += 1
                    continue
                start = y
                while y < y1 and mask[y, x]:
                    y += 1
                end = y
                top = rgb[start - 1, x] if start > 0 and not mask[start - 1, x] else None
                bottom = rgb[end, x] if end < rgb.shape[0] and not mask[end, x] else None
                if top is None and bottom is None:
                    top = bottom = baseline
                elif top is None:
                    top = bottom
                elif bottom is None:
                    bottom = top
                for row in range(start, end):
                    t = (row - start + 1) / (end - start + 1)
                    out[row, x] = np.asarray(top) * (1 - t) + np.asarray(bottom) * t
    return out


def restore_text_background(image: Image.Image, mask: np.ndarray, region,
                            method: str = 'telea', average: bool = False) -> Image.Image:
    """Replace ONLY approved mask pixels; no text/colour overlays are exported."""
    if method not in ('solid', 'horizontal', 'vertical', 'plane', 'quadratic', 'telea'):
        raise ValueError('Unknown background restoration method')
    rgb = np.asarray(image.convert('RGB'))
    if mask.shape != rgb.shape[:2]:
        raise ValueError('Mask and image dimensions differ')
    boolean_mask = mask > 0
    if not np.any(boolean_mask):
        return image.copy()
    bounds = _bounds(image, region)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return image.copy()
    valid = np.zeros(boolean_mask.shape, dtype=bool)
    x0, y0, x1, y1 = bounds
    valid[y0:y1, x0:x1] = True
    boolean_mask &= valid
    if not np.any(boolean_mask):
        return image.copy()
    baseline, _ = _reference_colour(rgb, boolean_mask, bounds)
    if method == 'solid':
        if baseline is None:
            baseline = np.mean(rgb[~boolean_mask], axis=0)
        filled = np.broadcast_to(baseline, rgb.shape).copy()
    elif method in ('horizontal', 'vertical'):
        filled = _fill_axis(rgb, boolean_mask, bounds,
                            horizontal=method == 'horizontal')
    else:
        repaired = repair_background(
            image, np.uint8(boolean_mask) * 255, method)
        filled = np.asarray(repaired.convert('RGB'), dtype=np.float32)

    if average:
        mean, samples = _reference_colour(rgb, boolean_mask, bounds, grid=True)
        if mean is not None:
            # Genuine background variation sets an adaptive permissible radius.
            scatter = np.linalg.norm(samples - mean, axis=1)
            threshold = max(32.0, float(np.percentile(scatter, 90)) + 15.0)
            edited = filled[boolean_mask]
            distance = np.linalg.norm(edited - mean, axis=1)
            factor = np.minimum(1.0, threshold / np.maximum(distance, 1e-5))
            filled[boolean_mask] = mean + (edited - mean) * factor[:, None]

    arr = np.asarray(image).copy()
    arr[boolean_mask, :3] = np.uint8(np.clip(np.rint(filled[boolean_mask]), 0, 255))
    return Image.fromarray(arr, image.mode)
