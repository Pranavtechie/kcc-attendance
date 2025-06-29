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

# --- GUI Application (PySide6) ---------------------------------------------
APP_ENROL_FRAMES = 20
APP_FRAME_WIDTH = 640
APP_FRAME_HEIGHT = 480
APP_TIMER_INTERVAL_MS = 33  # For ~30 FPS
