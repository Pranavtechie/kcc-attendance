import os
import sys

# Add rknn-toolkit2 path to python path
# The path is relative to this script's location
# The rknn_toolkit2 is expected to be installed, for example by:
# pip install rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# (adjust for your python version)
# If not installed, you can try uncommenting the following lines:
# script_dir = os.path.dirname(os.path.abspath(__file__))
# rknn_toolkit_path = os.path.join(script_dir, '..', 'rknn-toolkit2-v2.3.2-2025-04-09', 'rknn-toolkit2', 'packages')
# sys.path.insert(0, rknn_toolkit_path)

try:
    from rknn.api import RKNN
except ImportError:
    print("Error: rknn_toolkit2 is not installed or not in python path.")
    print(
        "Please install it from the provided .whl file in rknn-toolkit2-v2.3.2-2025-04-09/rknn-toolkit2/packages/..."
    )
    sys.exit(1)


def convert_model(onnx_model_path, rknn_model_path, target_platform="rk3588"):
    """
    Converts an ONNX model to a RKNN model.
    """
    print(f"--> Starting conversion for {onnx_model_path}")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print("--> Config model")
    # For models that take RGB images 0-255 as input
    rknn.config(
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=target_platform
    )
    print("done")

    # Load ONNX model
    print("--> Loading model")
    ret = rknn.load_onnx(model=onnx_model_path)
    if ret != 0:
        print("Load model failed!")
        rknn.release()
        return
    print("done")

    # Build model
    print("--> Building model")
    # Set do_quantization=False as we don't have a dataset for quantization
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Build model failed!")
        rknn.release()
        return
    print("done")

    # Export RKNN model
    print("--> Exporting rknn model")
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print("Export rknn model failed!")
        rknn.release()
        return
    print("done")

    # Release RKNN object
    rknn.release()
    print(f"--> Conversion successful: {rknn_model_path}")


if __name__ == "__main__":
    # Assuming the script is in the 'scripts' directory and 'models' is a sibling directory.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    # Models to convert
    models_to_convert = {"scrfd_2.5g.onnx": "scrfd_2.5g.rknn", "w600k_r50.onnx": "w600k_r50.rknn"}

    for onnx_name, rknn_name in models_to_convert.items():
        onnx_path = os.path.join(models_dir, onnx_name)
        rknn_path = os.path.join(models_dir, rknn_name)

        if not os.path.exists(onnx_path):
            print(f"Model not found: {onnx_path}")
            continue

        convert_model(onnx_path, rknn_path)

    print("\nAll model conversions finished.")
