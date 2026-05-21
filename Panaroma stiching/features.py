# Detector creation, keypoint detection, descriptor matching

import cv2
import numpy as np
from typing import Tuple, List


def make_detector():
    try:
        return cv2.SIFT_create()
    except Exception:
        return cv2.ORB_create(5000)


def detect_and_compute(detector, img_gray: np.ndarray):
    kp, des = detector.detectAndCompute(img_gray, None)
    return kp, des


def match_descriptors(des1, des2):
    if des1 is None or des2 is None:
        return []
    if des1.dtype == np.uint8 or des2.dtype == np.uint8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n[:2]
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good
