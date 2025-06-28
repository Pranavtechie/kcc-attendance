import onnxruntime as ort

from src.engine import FaceEngine  # run via `uv run -m src.test` from project root

print(ort.get_available_providers())
# ['CoreMLExecutionProvider', 'CPUExecutionProvider']   ← you should see this order

engine = FaceEngine()  # will pick CoreML automatically
