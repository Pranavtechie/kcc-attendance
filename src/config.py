"""Central configuration store.

All configuration variables should be defined here.
"""

from pathlib import Path

# --- Project paths ---------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Detector (SCRFD) ------------------------------------------------------
DETECTOR_MODEL_PATH = str(MODELS_DIR / "scrfd_2.5g.onnx")
DETECTOR_CONF_THRESHOLD = 0.5

# --- Recognizer (ArcFace) --------------------------------------------------
RECOGNIZER_MODEL_PATH = str(MODELS_DIR / "w600k_r50.onnx")
EMBEDDING_DIM = 512

# --- Face Database (FAISS + SQLite) ----------------------------------------
DB_PATH = str(DATA_DIR / "faces.db")
EMBEDDINGS_PATH = str(DATA_DIR / "embeddings.npy")
DB_SEARCH_THRESHOLD = 0.35

# --- RKNN models -----------------------------------------------------------
DETECTOR_RKNN_MODEL_PATH = str(MODELS_DIR / "rknn-weights" / "scrfd_2.5g.rknn")
RECOGNIZER_RKNN_MODEL_PATH = str(MODELS_DIR / "rknn-weights" / "w600k_r50.rknn")

# --- Inference Engine Selection --------------------------------------------
try:
    # rknnlite is the library for on-device inference
    from rknnlite.api import RKNNLite  # noqa: F401

    USE_RKNN = True
except (ImportError, ModuleNotFoundError):
    USE_RKNN = False

# --- GUI Application (PySide6) ---------------------------------------------
APP_ENROL_FRAMES = 20
APP_FRAME_WIDTH = 640
APP_FRAME_HEIGHT = 480
APP_TIMER_INTERVAL_MS = 33  # For ~30 FPS

APP_HIBERNATE_INTERVAL_MS = 5000  # Check for wake-up every 1 second
APP_BLACK_FRAME_THRESHOLD = 5.0   # Avg pixel value below which frame is considered black