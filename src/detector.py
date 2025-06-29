from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

from . import config
from .utils import nms

if config.USE_RKNN:
    from rknnlite.api import RKNNLite


class Face:
    """Detected face container"""

    def __init__(self, bbox: np.ndarray, score: float, kps: np.ndarray):
        self.bbox = bbox  # (4,) x1,y1,x2,y2
        self.score = score  # float
        self.kps = kps  # (5,2)

    def __repr__(self):
        return f"Face(score={self.score:.2f}, bbox={self.bbox.tolist()})"


class SCRFDDetector:
    """Lightweight SCRFD detector using ONNXRuntime (dynamic input supported).

    Only the stride set {8,16,32} is handled which is the case for the published
    scrfd_2.5g & 10g models.
    """

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.use_rknn = config.USE_RKNN

        if self.use_rknn:
            rknn_model_path = config.DETECTOR_RKNN_MODEL_PATH
            self.rknn = RKNNLite(verbose=False)
            ret = self.rknn.load_rknn(rknn_model_path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model: {rknn_model_path}")
            # NPU_CORE_0_1_2 is for RK3588
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                raise RuntimeError("Failed to init RKNN runtime")
            self.strides = (8, 16, 32)
        else:
            self.model_path = str(model_path)
            providers = (
                ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                if "CoreMLExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            # Assume three strides
            self.strides = (8, 16, 32)

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """Return chw float32 normalized image"""
        h0, w0 = img.shape[:2]
        # pad to square
        size = max(h0, w0)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[:h0, :w0] = img
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img -= 127.5
        img *= 1 / 128.0
        img = np.transpose(img, (2, 0, 1))[None]
        return img

    def __call__(self, img_bgr: np.ndarray) -> List[Face]:
        if self.use_rknn:
            # For RKNN, preprocessing is typically simpler (e.g., uint8 NHWC)
            h0, w0 = img_bgr.shape[:2]
            size = max(h0, w0)
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            padded[:h0, :w0] = img_bgr
            rknn_input = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            # Post-processing logic still relies on the original blob, so we create it.
            blob = self._preprocess(img_bgr)
            # RKNN inference needs a 4D input (N, H, W, C).
            rknn_input = np.expand_dims(rknn_input, axis=0)
            outs = self.rknn.inference(inputs=[rknn_input])
        else:
            blob = self._preprocess(img_bgr)
            outs = self.session.run(self.output_names, {self.input_name: blob})
        # unpack outputs
        scores_list = outs[0:3]
        bboxes_list = outs[3:6]
        kps_list = outs[6:9]
        faces: List[Face] = []
        for stride, scores, bbox_preds, kps_preds in zip(
            self.strides, scores_list, bboxes_list, kps_list
        ):
            # scores shape (N,1) -> (N,)
            scores = scores.squeeze(1)
            keep_inds = np.where(scores > self.conf_threshold)[0]
            if keep_inds.size == 0:
                continue
            scores = scores[keep_inds]
            bbox_preds = bbox_preds[keep_inds] * stride
            kps_preds = kps_preds[keep_inds] * stride
            # SCRFD uses two anchors per spatial cell for all published models
            anchor_num = 2

            # The padded input fed to the network is a square whose side is `size`
            # (see _preprocess). Therefore, the feature-map height and width can
            # be computed directly from the stride.
            input_size = blob.shape[2]  # square side length after padding
            fmap_h = fmap_w = int(input_size / stride)

            # Safety ‑ some ONNX runtimes may pad the output so N can be larger
            # than fmap_h * fmap_w * anchor_num by zero-padding.  We therefore
            # extend the grid until it is large enough.
            while fmap_h * fmap_w * anchor_num < scores.shape[0]:
                fmap_h += 1
                fmap_w += 1

            shifts_x = np.tile(np.arange(fmap_w), (fmap_h, 1)).flatten()
            shifts_y = np.tile(np.arange(fmap_h)[:, None], (1, fmap_w)).flatten()
            anchor_centers = np.stack((shifts_x, shifts_y), axis=1)
            anchor_centers = np.repeat(anchor_centers, anchor_num, axis=0)
            anchor_centers = (anchor_centers + 0.5) * stride
            centers = anchor_centers[keep_inds]
            # decode bbox
            x1y1 = centers - bbox_preds[:, 0:2]
            x2y2 = centers + bbox_preds[:, 2:4]
            bboxes = np.hstack([x1y1, x2y2])
            # decode kps
            kps = kps_preds.reshape((-1, 5, 2)) + centers[:, None, :]
            for b, s, kp in zip(bboxes, scores, kps):
                # No scaling needed as preprocess only pads, not resizes.
                faces.append(Face(bbox=b, score=float(s), kps=kp))
        if not faces:
            return []
        # NMS
        boxes = np.vstack([f.bbox for f in faces])
        scrs = np.array([f.score for f in faces])
        keep = nms(boxes, scrs, 0.4)
        faces = [faces[i] for i in keep]
        return faces
