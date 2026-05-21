# Seam finding and multiband blending

import cv2
import numpy as np
from scipy import ndimage
from typing import Optional, Tuple


def compute_overlap_cost(imgA_gray, imgB_gray):
    gradA = np.sqrt(cv2.Sobel(imgA_gray, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(imgA_gray, cv2.CV_32F, 0, 1, ksize=3)**2)
    gradB = np.sqrt(cv2.Sobel(imgB_gray, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(imgB_gray, cv2.CV_32F, 0, 1, ksize=3)**2)
    cost = gradA + gradB
    return cost


def find_vertical_seam(overlapA, overlapB, maskA, maskB) -> Optional[np.ndarray]:
    valid = (maskA > 0) & (maskB > 0)
    if valid.sum() == 0:
        return None
    ys, xs = np.where(valid)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    A_crop = overlapA[y0:y1 + 1, x0:x1 + 1]
    B_crop = overlapB[y0:y1 + 1, x0:x1 + 1]
    A_gray = cv2.cvtColor(A_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    B_gray = cv2.cvtColor(B_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(A_gray - B_gray)
    cost_img = compute_overlap_cost(A_gray, B_gray) + diff
    h, w = cost_img.shape
    M = np.full((h, w), np.inf, dtype=np.float64)
    backtrack = np.zeros((h, w), dtype=np.int32)
    M[0, :] = cost_img[0, :]
    for i in range(1, h):
        for j in range(w):
            best = M[i - 1, j]
            arg = j
            if j > 0 and M[i - 1, j - 1] < best:
                best = M[i - 1, j - 1]
                arg = j - 1
            if j < w - 1 and M[i - 1, j + 1] < best:
                best = M[i - 1, j + 1]
                arg = j + 1
            M[i, j] = cost_img[i, j] + best
            backtrack[i, j] = arg
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(M[-1])
    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]
    seam_x = seam + x0
    seam_y = np.arange(y0, y1 + 1)
    path = np.stack([seam_y, seam_x], axis=1)
    return path


def make_seam_mask(canvas_shape: Tuple[int, int, int], seam_path: np.ndarray, left_side='left'):
    h, w = canvas_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    ys = seam_path[:, 0]
    xs = seam_path[:, 1]
    for (y, x) in zip(ys, xs):
        if left_side == 'left':
            mask[y, :x] = 1
        else:
            mask[y, x:] = 1
    return mask


def gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img.astype(np.float32))
    return gp


def laplacian_pyramid(img, levels):
    gp = gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        size = (gp[i].shape[1], gp[i].shape[0])
        GE = cv2.pyrUp(gp[i + 1], dstsize=size)
        L = gp[i] - GE
        lp.append(L)
    lp.append(gp[-1])
    return lp


def reconstruct_from_laplacian(lp):
    levels = len(lp) - 1
    img = lp[-1]
    for i in range(levels - 1, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + lp[i]
    return img


def multiband_blend(imgA, imgB, mask, levels=5):
    mask = mask.astype(np.float32)
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    lpA = laplacian_pyramid(imgA, levels)
    lpB = laplacian_pyramid(imgB, levels)
    gpM = gaussian_pyramid(mask, levels)
    blended_pyr = []
    for LA, LB, GM in zip(lpA, lpB, gpM):
        blended = GM * LA + (1.0 - GM) * LB
        blended_pyr.append(blended)
    blended = reconstruct_from_laplacian(blended_pyr)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended