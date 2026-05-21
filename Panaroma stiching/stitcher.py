# High level stitching pipeline that imports modular pieces

import cv2
import numpy as np
from typing import List, Optional
from utils import read_images, gray
from projection import cylindrical_project
from features import make_detector, detect_and_compute, match_descriptors
from homography import find_homography, accumulate_homographies
from warper import estimate_canvas_size, warp_to_canvas
from exposure import global_exposure_compensation
from seam_blend import find_vertical_seam, make_seam_mask, multiband_blend
from scipy import ndimage


def stitch_images(images: List[np.ndarray], focal: Optional[float] = None, auto_cyl: bool = True) -> np.ndarray:
    n = len(images)
    if auto_cyl:
        if focal is None:
            focal = images[0].shape[1]
        cyl_imgs = [cylindrical_project(img, focal) for img in images]
    else:
        cyl_imgs = images

    detector = make_detector()
    kps = []
    dess = []
    for img in cyl_imgs:
        g = gray(img)
        kp, des = detect_and_compute(detector, g)
        kps.append(kp)
        dess.append(des)

    homographies = []
    matches_list = []
    for i in range(n - 1):
        matches = match_descriptors(dess[i], dess[i + 1])
        print(f"Pair {i}->{i+1} keypoints: {len(kps[i])},{len(kps[i+1])} matches: {len(matches)}")
        try:
            top = sorted(matches, key=lambda m: m.distance)[:50]
            debug_img = cv2.drawMatches(cyl_imgs[i], kps[i], cyl_imgs[i+1], kps[i+1], top, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            debug_name = f"matches_{i}_{i+1}.jpg"
            cv2.imwrite(debug_name, debug_img)
            print(f"Saved {debug_name}")
        except Exception as e:
            print("Could not draw matches", e)

        H, mask = find_homography(kps[i], kps[i + 1], matches, ransac_thresh=5.0)
        if H is None:
            H, mask = find_homography(kps[i], kps[i + 1], matches, ransac_thresh=8.0)
        if H is None:
            H, mask = find_homography(kps[i], kps[i + 1], matches, ransac_thresh=12.0)

        if H is None and len(matches) >= 4:
            pts1 = np.float32([kps[i][m.queryIdx].pt for m in matches])
            pts2 = np.float32([kps[i+1][m.trainIdx].pt for m in matches])
            try:
                A, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=8.0)
                if A is not None:
                    H = np.eye(3)
                    H[:2, :] = A
                    print(f"Affine fallback used for pair {i}->{i+1}")
            except Exception as e:
                print("Affine fallback failed", e)

        if H is None:
            raise RuntimeError(f"Could not estimate homography between {i} and {i+1}. Check matches_{i}_{i+1}.jpg")

        homographies.append(H)
        matches_list.append(matches)

    Htot = accumulate_homographies(homographies)
    canvas_w, canvas_h, offset = estimate_canvas_size(cyl_imgs, Htot)

    warped_imgs = []
    masks = []
    for img, H in zip(cyl_imgs, Htot):
        wimg, m = warp_to_canvas(img, H, canvas_w, canvas_h, offset)
        warped_imgs.append(wimg)
        masks.append(m)

    warped_imgs = global_exposure_compensation(warped_imgs, masks)

    base_img = warped_imgs[0].copy()
    base_mask = masks[0].copy()

    for i in range(1, n):
        A = base_img
        B = warped_imgs[i]
        maskA = base_mask
        maskB = masks[i]
        union_mask = ((maskA > 0) | (maskB > 0)).astype(np.uint8)
        overlap = (maskA > 0) & (maskB > 0)
        if overlap.sum() > 0:
            seam_path = find_vertical_seam(A, B, maskA, maskB)
            if seam_path is not None:
                seam_maskA = make_seam_mask(A.shape, seam_path, left_side='left')
            else:
                seam_maskA = (maskA > 0).astype(np.uint8)
        else:
            seam_maskA = (maskA > 0).astype(np.uint8)

        mask_blend = ndimage.gaussian_filter(seam_maskA.astype(np.float32), sigma=5)
        denom = (mask_blend.max() - mask_blend.min() + 1e-8)
        mask_blend = (mask_blend - mask_blend.min()) / denom

        ys, xs = np.where(union_mask > 0)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        A_crop = A[y0:y1 + 1, x0:x1 + 1]
        B_crop = B[y0:y1 + 1, x0:x1 + 1]
        M_crop = mask_blend[y0:y1 + 1, x0:x1 + 1]

        try:
            blended = multiband_blend(A_crop, B_crop, M_crop, levels=5)
        except Exception:
            blended = (A_crop * M_crop[:, :, None] + B_crop * (1 - M_crop)[:, :, None]).astype(np.uint8)

        base_img[y0:y1 + 1, x0:x1 + 1] = blended
        base_mask = ((base_mask > 0) | (maskB > 0)).astype(np.uint8)

    gray_final = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_final, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = base_img[y:y + h, x:x + w]
    else:
        cropped = base_img

    h_final, w_final = cropped.shape[:2]
    if h_final > w_final:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_final, w_final = cropped.shape[:2]
    max_width = 1600
    if w_final > max_width:
        scale = max_width / w_final
        cropped = cv2.resize(cropped, (int(w_final * scale), int(h_final * scale)), interpolation=cv2.INTER_AREA)

    try:
        gray_pano = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_pano, 80, 160)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
        tilt = 0.0
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180.0 / np.pi
                if 85.0 < angle < 95.0 or -95.0 < angle < -85.0:
                    angles.append(angle)
            if len(angles) > 0:
                tilt = float(np.mean(angles) - 90.0)
        if abs(tilt) > 0.01:
            M = cv2.getRotationMatrix2D((cropped.shape[1] // 2, cropped.shape[0] // 2), -tilt, 1.0)
            straight = cv2.warpAffine(cropped, M, (cropped.shape[1], cropped.shape[0]), borderValue=(0, 0, 0))
            gray_straight = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray_straight, 5, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                straight = straight[y:y + h, x:x + w]
                cropped = straight
    except Exception:
        pass

    return cropped