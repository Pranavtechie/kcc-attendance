import onnx


def inspect_onnx_model(model_path):
    try:
        model = onnx.load(model_path)
        print(f"Model: {model_path}")
        for i, input_tensor in enumerate(model.graph.input):
            input_name = input_tensor.name
            input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  Input {i}:")
            print(f"    Name: {input_name}")
            print(f"    Shape: {input_shape} (batch_size, channels, height, width)")
        print("-" * 30)
    except Exception as e:
        print(f"Error inspecting {model_path}: {e}")


if __name__ == "__main__":
    models_to_inspect = [
        "models/scrfd_2.5g.onnx",
        "models/w600k_r50.onnx",
    ]
    for model_path in models_to_inspect:
        inspect_onnx_model(model_path)
