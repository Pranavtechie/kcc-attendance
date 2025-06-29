from rknnlite.api import RKNNLite


def inspect_rknn_model(model_path):
    rknn = None
    try:
        rknn = RKNNLite()
        print(f"Model: {model_path}")

        # Load RKNN model
        print(f"  Loading RKNN model: {model_path}")
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            print(f"  Error loading RKNN model: {model_path}, ret={ret}")
            return

        # Query model info
        print("  Querying model info...")
        info = rknn.query_model_info()
        if info:
            print("  Inputs:")
            for i, input_info in enumerate(info["inputs"]):
                print(f"    Input {i}:")
                print(f"      Name: {input_info['name']}")
                print(f"      Shape: {input_info['shape']}")
                print(f"      Dtype: {input_info['dtype']}")
                print(f"      Quantization: {input_info['qnt_type']}")

            print("  Outputs:")
            for i, output_info in enumerate(info["outputs"]):
                print(f"    Output {i}:")
                print(f"      Name: {output_info['name']}")
                print(f"      Shape: {output_info['shape']}")
                print(f"      Dtype: {output_info['dtype']}")
                print(f"      Quantization: {output_info['qnt_type']}")
        else:
            print("  Failed to query model info.")

        print("-" * 30)

    except Exception as e:
        print(f"Error inspecting {model_path}: {e}")
    finally:
        if rknn:
            rknn.release()


if __name__ == "__main__":
    models_to_inspect = [
        "models/scrfd_2.5g.rknn",
        "models/w600k_r50.rknn",
    ]
    for model_path in models_to_inspect:
        inspect_rknn_model(model_path)
