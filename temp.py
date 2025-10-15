import tf2onnx
import tensorflow as tf

input_sig = (tf.TensorSpec([1, 224, 224, 3], tf.float32, name="images"),)
output_path = "mobilenet_v3_small.onnx"

# Works in tf2onnx ≥1.9
tf2onnx.convert.from_saved_model(
    "path/to/saved_model",
    input_signature=input_sig,
    opset=13,
    output_path=output_path,
)
print("✅ ONNX exported:", output_path)


from tf2onnx import convert

convert.from_saved_model(
    "path/to/saved_model",
    input_signature=input_sig,
    opset=13,
    output_path="mobilenet_v3_small.onnx"
)


!python -m onnxruntime.tools.print_onnx_info mobilenet_v3_small.onnx
