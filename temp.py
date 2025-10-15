import tensorflow as tf
model = tf.saved_model.load("path/to/saved_model")
sig = model.signatures["serving_default"]

for name, tensor in sig.structured_input_signature[1].items():
    print("Input:", name, tensor.shape, tensor.dtype)

for name, tensor in sig.structured_outputs.items():
    print("Output:", name, tensor.shape, tensor.dtype)
