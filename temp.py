import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # skip GPU init
import tensorflow as tf

model = tf.saved_model.load("path/to/saved_model")
sig = model.signatures["serving_default"]  # may still take a bit, but faster now

for n, t in sig.structured_input_signature[1].items():
    print("Input:", n, t.shape, t.dtype)
for n, t in sig.structured_outputs.items():
    print("Output:", n, t.shape, t.dtype)
