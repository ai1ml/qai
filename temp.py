import sys, inspect, pkgutil
import tf2onnx, tensorflow as tf
print("tf2onnx version:", getattr(tf2onnx, "__version__", "unknown"))
print("tf2onnx path:", inspect.getfile(tf2onnx))
print("tensorflow version:", tf.__version__)

SAVEDMODEL_DIR = "path/to/saved_model"   # folder with saved_model.pb
ONNX_PATH = "mobilenet_v3_small.onnx"
IMG = 224

# If your SavedModel already has a numeric input signature you want to enforce:
!python -m tf2onnx.convert \
  --saved-model "$SAVEDMODEL_DIR" \
  --signature_def "serving_default" \
  --inputs "images:0" \
  --inputs-as-nchw "" \
  --outputs "logits:0" \
  --opset 13 \
  --output "$ONNX_PATH"

import tensorflow as tf
from tf2onnx import convert

SAVEDMODEL_DIR = "path/to/saved_model"
IMG = 224
ONNX_PATH = "mobilenet_v3_small.onnx"

loaded = tf.saved_model.load(SAVEDMODEL_DIR)
concrete = loaded.signatures["serving_default"]   # must be numeric (float32), not DT_STRING

spec = (tf.TensorSpec([1, IMG, IMG, 3], tf.float32, name="images"),)
model_proto, _ = convert.from_function(
    concrete, input_signature=spec, opset=13, output_path=ONNX_PATH
)
print("✅ ONNX exported:", ONNX_PATH)





import tensorflow as tf
from tf2onnx.convert import from_keras

keras_model = tf.keras.models.load_model("path/to/saved_model")  # works if it’s a Keras export
spec = (tf.TensorSpec([1,224,224,3], tf.float32, name="images"),)
model_proto, _ = from_keras(keras_model, input_signature=spec, opset=13, output_path="mobilenet_v3_small.onnx")




import pip, sys
!pip install -U "tensorflow==2.12.*" "tf2onnx==1.15.1"



import qai_hub as hub
hub.login(api_token="<YOUR_QAI_HUB_API_TOKEN>")

device = hub.Device("Snapdragon X Elite CRD")   # or your target
job = hub.submit_compile_job(
    model="mobilenet_v3_small.onnx",
    device=device,
    name="mnetv3_small_fp16",
    input_specs={"images": ((1,224,224,3), "float32")},  # NHWC FP32
    options="--precision fp16",
)
print("Compile job id:", job.id)
# For INT8, add: calibration_data="./calib_images" and options="--precision int8"
