import sys
from typing import Tuple

import cv2
import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> np.ndarray:
    """Non-maximum suppression.

    Args:
        boxes: (N, 4) array of x1, y1, x2, y2.
        scores: (N,) confidence scores.
        iou_threshold: IoU threshold.
    Returns:
        indices of kept boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


def five_point_align(
    src_img: np.ndarray, src_pts: np.ndarray, size: int = 112
) -> Tuple[np.ndarray, np.ndarray]:
    """Align face using 5 landmarks.

    Args:
        src_img: BGR image.
        src_pts: (5,2) landmarks in original image.
        size: output image size.
    Returns:
        aligned RGB image, 2×3 affine matrix.
    """
    dst_pts = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    dst_pts *= size / 112.0
    src_pts = src_pts.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    aligned = cv2.warpAffine(src_img, M, (size, size))
    return aligned, M


def find_available_camera(max_index: int = 5) -> int:
    """Finds the first available camera index.

    Tries to open cameras from index 0 to `max_index`.
    The first one that can be opened and provides a frame is returned.
    If no camera is found, the program exits.

    Args:
        max_index: Maximum camera index to check.

    Returns:
        The first available camera index.
    """
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        ret, _ = cap.read()
        cap.release()
        if ret:
            print(f"Found available camera at index {i}")
            return i
    print(f"No available camera found (checked 0 to {max_index}). Exiting.")
    sys.exit(1)
