import cv2
import numpy as np
from typing import List, Tuple


def estimate_canvas_size(images: List[np.ndarray], Htot: List[np.ndarray]) -> Tuple[int, int, Tuple[int, int]]:
    corners = []
    for img, H in zip(images, Htot):
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        pts_ = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H)
        corners.append(pts_.reshape(-1, 2))
    all_pts = np.vstack(corners)
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    offset = (-x_min, -y_min)
    return canvas_w, canvas_h, offset


def warp_to_canvas(img: np.ndarray, H: np.ndarray, canvas_w: int, canvas_h: int, offset: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    ox, oy = offset
    H_off = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)
    Htotal = H_off @ H
    canvas_img = cv2.warpPerspective(img, Htotal, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.ones(img.shape[:2], dtype=np.uint8) * 255, Htotal, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    mask = (mask > 0).astype(np.uint8)
    return canvas_img, mask