import numpy as np


def crop_data(depth, mask, bbox):
    x1, y1, x2, y2 = bbox

    depth = depth[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]

    return depth, mask


def get_bbox_from_mask(mask):
    mask_pixels = np.where(mask > 0)

    if len(mask_pixels[0]) == 0:
        return -1, -1, -1, -1

    bbox_x1 = int(np.min(mask_pixels[1]))
    bbox_y1 = int(np.min(mask_pixels[0]))
    bbox_x2 = int(np.max(mask_pixels[1]))
    bbox_y2 = int(np.max(mask_pixels[0]))

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2