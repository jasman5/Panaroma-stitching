import numpy as np
import cv2
from typing import List, Tuple


def find_homography(kp1, kp2, matches, ransac_thresh=5.0):
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    return H, mask


def accumulate_homographies(homographies: List[np.ndarray]) -> List[np.ndarray]:
    n = len(homographies) + 1
    F = [np.eye(3)]
    for i in range(len(homographies)):
        F.append(homographies[i] @ F[-1])
    center = n // 2
    F_center = F[center]
    Htot = []
    for i in range(n):
        H_i = np.linalg.inv(F[i]) @ F_center
        Htot.append(H_i / H_i[2, 2])
    return Htot