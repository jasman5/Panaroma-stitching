# Cylindrical projection

import numpy as np
from typing import Tuple


def cylindrical_project(img: np.ndarray, f: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    result = np.zeros_like(img)

    xs = np.arange(w)
    ys = np.arange(h)
    xs_grid, ys_grid = np.meshgrid(xs, ys)

    x = xs_grid - cx
    y = ys_grid - cy

    theta = np.arctan2(x, f)
    h_ = y / np.sqrt(x * x + f * f)

    x_proj = f * np.tan(theta)
    y_proj = h_ * np.sqrt(x_proj * x_proj + f * f)

    x_src = x_proj + cx
    y_src = y_proj + cy

    valid = (x_src >= 0) & (x_src < w - 1) & (y_src >= 0) & (y_src < h - 1)

    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)

    x0_clipped = np.clip(x0, 0, w - 2)
    y0_clipped = np.clip(y0, 0, h - 2)
    x1_clipped = x0_clipped + 1
    y1_clipped = y0_clipped + 1

    xf = x_src - x0
    yf = y_src - y0

    wa = (1 - xf) * (1 - yf)
    wb = xf * (1 - yf)
    wc = (1 - xf) * yf
    wd = xf * yf

    for c in range(img.shape[2]):
        channel = img[:, :, c]
        sampled = (
            wa * channel[y0_clipped, x0_clipped] +
            wb * channel[y0_clipped, x1_clipped] +
            wc * channel[y1_clipped, x0_clipped] +
            wd * channel[y1_clipped, x1_clipped]
        )
        sampled[~valid] = 0
        result[:, :, c] = sampled

    return result