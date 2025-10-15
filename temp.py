# 0) installs (first run only)
# !pip install -q qai-hub tf2onnx onnx tensorflow

import os
import tensorflow as tf
import tf2onnx
import qai_hub as hub

API_TOKEN = "<YOUR_QAI_HUB_API_TOKEN>"
SAVEDMODEL_DIR = "path/to/saved_model"   # folder with saved_model.pb + variables/
ONNX_PATH = "mobilenet_v3_small.onnx"
IMG_SIZE = 224
NUM_CLASSES = 5
TARGET_DEVICE_NAME = "Snapdragon X Elite CRD"   # e.g., "Snapdragon X Elite CRD", "RB6", etc.

# 1) Convert SavedModel -> ONNX with a fixed, numeric input signature
#    (If your SavedModel had bytes input, ensure this signature matches the numeric path.)
input_sig = (tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="images"),)

model_proto, _ = tf2onnx.convert.from_saved_model(
    SAVEDMODEL_DIR,
    input_signature=input_sig,
    opset=13,
    output_path=ONNX_PATH,
)

print("ONNX saved to:", ONNX_PATH)

# 2) Login and pick target device
hub.login(api_token=API_TOKEN)
device = hub.Device(TARGET_DEVICE_NAME)

# 3A) Compile FP16 (quickest smoke test)
job_fp16 = hub.submit_compile_job(
    model=ONNX_PATH,
    device=device,
    name="mobilenetv3_small_fp16",
    input_specs={"images": ((1, IMG_SIZE, IMG_SIZE, 3), "float32")},  # NHWC FP32
    options="--precision fp16",
)
compiled_fp16 = job_fp16.get_target_model()   # .dlc artifact handle
print("FP16 compile submitted; job id:", job_fp16.id)

# 3B) (Optional) Compile INT8 with calibration images (raw or preprocessed same as runtime)
# Put ~50–200 representative images into ./calib_images (jpg/png)
CALIB_DIR = "./calib_images"
if os.path.isdir(CALIB_DIR):
    job_int8 = hub.submit_compile_job(
        model=ONNX_PATH,
        device=device,
        name="mobilenetv3_small_int8",
        input_specs={"images": ((1, IMG_SIZE, IMG_SIZE, 3), "float32")},
        calibration_data=CALIB_DIR,           # or an uploaded dataset ID
        options="--precision int8",
    )
    compiled_int8 = job_int8.get_target_model()
    print("INT8 compile submitted; job id:", job_int8.id)
