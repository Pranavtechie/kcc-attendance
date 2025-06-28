from pathlib import Path
from typing import List, Tuple

import numpy as np

# Prefer absolute import so that the module works both when imported as
# part of the `src` package (``import src.engine``) **and** when the file is
# executed/loaded as a top-level script (which gives it the name ``engine``).
from . import config
from .detector import Face, SCRFDDetector
from .recognizer import ArcFaceRecognizer


class FaceEngine:
    """High-level façade replicating InspireFaceSession (detect + extract)."""

    def __init__(
        self,
        detector_model: str | Path = config.DETECTOR_MODEL_PATH,
        recognizer_model: str | Path = config.RECOGNIZER_MODEL_PATH,
        conf_threshold: float = config.DETECTOR_CONF_THRESHOLD,
    ) -> None:
        self.detector = SCRFDDetector(detector_model, conf_threshold)
        self.recognizer = ArcFaceRecognizer(recognizer_model)

    # --- InspireFace-like API -------------------------------------------------

    def face_detection(self, frame_bgr: np.ndarray) -> List[Face]:
        return self.detector(frame_bgr)

    def face_feature_extract(self, frame_bgr: np.ndarray, face: Face) -> np.ndarray | None:
        if face is None:
            return None
        return self.recognizer.get(frame_bgr, face.kps)

    # convenience ------------------------------------------------------------
    def detect_and_extract(
        self, frame_bgr: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Face, np.ndarray]]:
        faces = self.face_detection(frame_bgr)
        faces.sort(key=lambda f: f.score, reverse=True)
        faces = faces[:top_k]
        feats = [self.face_feature_extract(frame_bgr, f) for f in faces]
        return list(zip(faces, feats))
