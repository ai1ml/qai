# pip install qai-hub tf2onnx onnx
import os, qai_hub as hub

# 0) Auth once (or set env QAIHUB_API_TOKEN)
hub.login(api_token="<YOUR_TOKEN>")

# 1) Convert TF SavedModel -> ONNX (only once per model build)
#    From a shell (recommended):
#    python -m tf2onnx.convert --saved-model ./saved_model --output mobilenet_v3_small.onnx --opset 13

onnx_path = "mobilenet_v3_small.onnx"   # the file created above

# 2) Pick a device target
device = hub.Device("Samsung Galaxy S23")  # or hub.get_devices()[0], etc.  [oai_citation:1‡AI Hub](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_compile_job.html)

# 3) (Optional) Overwrite input name/shape/dtype if needed
#    For ONNX, input_specs are optional, but you can pin static shape & dtype:
input_specs = {"images": ((1, 224, 224, 3), "float32")}  # NHWC FP32 for TF-origin models  [oai_citation:2‡AI Hub](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_compile_job.html)

# 4A) Compile FP16 (quick smoke test)
job_fp16 = hub.submit_compile_job(
    model=onnx_path,
    device=device,
    name="mnetv3-small-fp16",
    input_specs=input_specs,
    options="--precision fp16",
)
target_model_fp16 = job_fp16.get_target_model()  # .dlc holder

# 4B) Compile INT8 (with calibration images folder)
#    Provide a folder of raw images AFTER the same preprocessing your runtime will do.
job_int8 = hub.submit_compile_job(
    model=onnx_path,
    device=device,
    name="mnetv3-small-int8",
    input_specs=input_specs,
    calibration_data="./calibration_images",  # path or uploaded Dataset ID
    options="--precision int8",
)
target_model_int8 = job_int8.get_target_model()
