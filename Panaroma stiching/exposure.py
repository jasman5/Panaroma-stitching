import numpy as np


def exposure_compensate_pair(img1, mask1, img2, mask2):
    overlap = (mask1 > 0) & (mask2 > 0)
    if overlap.sum() < 100:
        return img1, img2
    means1 = [img1[:, :, c][overlap].mean() for c in range(3)]
    means2 = [img2[:, :, c][overlap].mean() for c in range(3)]
    gains = [m1 / (m2 + 1e-8) for m1, m2 in zip(means1, means2)]
    img2_adj = img2.astype(np.float32).copy()
    for c in range(3):
        img2_adj[:, :, c] = img2_adj[:, :, c] * gains[c]
    np.clip(img2_adj, 0, 255, out=img2_adj)
    return img1, img2_adj.astype(np.uint8)


def global_exposure_compensation(warped_imgs, masks):
    base = warped_imgs[0].astype(np.float32)
    base_mask = masks[0]
    compensated = [warped_imgs[0]]
    for i in range(1, len(warped_imgs)):
        img = warped_imgs[i].astype(np.uint8)
        mask = masks[i]
        _, img_adj = exposure_compensate_pair(base, base_mask, img, mask)
        compensated.append(img_adj)
    return compensated