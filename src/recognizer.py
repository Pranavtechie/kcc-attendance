from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from . import config
from .utils import five_point_align

if config.USE_RKNN:
    from rknnlite.api import RKNNLite


class ArcFaceRecognizer:
    """ONNX ArcFace / Glint360K recognizer."""

    def __init__(
        self,
        model_path: str | Path = config.RECOGNIZER_MODEL_PATH,
        embed_size: int = config.EMBEDDING_DIM,
    ):
        self.embed_size = embed_size
        self.use_rknn = config.USE_RKNN

        if self.use_rknn:
            rknn_model_path = config.RECOGNIZER_RKNN_MODEL_PATH
            self.rknn = RKNNLite(verbose=False)
            ret = self.rknn.load_rknn(rknn_model_path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model: {rknn_model_path}")
            # NPU_CORE_0_1_2 is for RK3588
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                raise RuntimeError("Failed to init RKNN runtime")
        else:
            providers = (
                ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                if "CoreMLExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name

    def get(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Return L2-normalised 512-d embedding.

        Args:
            img_bgr: original frame.
            kps: (5,2) landmarks corresponding to face.
        """
        aligned, _ = five_point_align(img_bgr, kps, 112)

        if self.use_rknn:
            # RKNN model expects uint8 NHWC input.
            face = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            face = face.astype(np.float32) / 255.0
            face = (face - 0.5) / 0.5
            face = np.transpose(face, (2, 0, 1))[None]
            # RKNN inference needs a 4D input (N, H, W, C).
            emb = self.rknn.inference(inputs=[face])[0][0]
        else:
            face = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            face = face.astype(np.float32) / 255.0
            face = (face - 0.5) / 0.5
            face = np.transpose(face, (2, 0, 1))[None]
            emb = self.session.run(None, {self.input_name: face})[0][0]

        emb = emb / np.linalg.norm(emb)
        return emb.astype(np.float32)
