# General utilities: I/O and tiny helpers.

import cv2
import numpy as np
from typing import List
import os


def read_images(paths: List[str]) -> List[np.ndarray]:
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read {p}")
        imgs.append(img)
    return imgs


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)