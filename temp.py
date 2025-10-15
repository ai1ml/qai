S3_MODEL_ARTIFACT = "s3://<bucket>/<path>/model.tar.gz"   # <-- your JumpStart training output
TARGET_DEVICE_NAME = "Snapdragon X Elite CRD"             # e.g., "Snapdragon X Elite CRD", "RB6", "Snapdragon 8 Gen 3"
IMG_SIZE = 224                                            # your model input size
PRECISION = "fp16"                                        # "fp16" or "int8"
S3_CALIB_IMAGES = None                                    # e.g., "s3://<bucket>/calib_images/" for INT8, else None
AI_HUB_API_TOKEN = "<YOUR_QAI_HUB_API_TOKEN>"             # put your token here
JOB_NAME_PREFIX = "tf-savedmodel-to-onnx-compile"
-------
%%writefile src/convert_and_compile.py
import os, io, sys, tarfile, argparse, json, shutil, tempfile, boto3
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # keep TF on CPU for reliability

def download_s3_to(path, s3_uri):
    s3 = boto3.client("s3")
    assert s3_uri.startswith("s3://")
    _, _, bucket, *rest = s3_uri.split("/", 3)
    key = rest[0] if rest else ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s3.download_file(bucket, key, path)

def sync_s3_dir_to_local(s3_uri, local_dir):
    if not s3_uri: return
    s3 = boto3.resource("s3")
    assert s3_uri.startswith("s3://")
    _, _, bucket, *rest = s3_uri.split("/", 3)
    key_prefix = rest[0] if rest else ""
    bucket_obj = s3.Bucket(bucket)
    for obj in bucket_obj.objects.filter(Prefix=key_prefix):
        rel = obj.key[len(key_prefix):].lstrip("/")
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        bucket_obj.download_file(obj.key, dest)

def find_saved_model_dir(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if "saved_model.pb" in filenames:
            return dirpath
    return None

def list_signature(saved_model_dir):
    # lightweight TF import inside function
    import tensorflow as tf
    m = tf.saved_model.load(saved_model_dir)
    sigs = list(m.signatures.keys())
    d = {}
    for k in sigs:
        fn = m.signatures[k]
        ins = {n: (t.dtype.name, tuple(t.shape)) for n, t in fn.structured_input_signature[1].items()}
        outs = {n: (t.dtype.name, tuple(t.shape)) for n, t in fn.structured_outputs.items()}
        d[k] = {"inputs": ins, "outputs": outs}
    return d

def convert_to_onnx(saved_model_dir, onnx_path, img_size, input_name="images", signature="serving_default"):
    import tensorflow as tf
    from tf2onnx.convert import from_saved_model
    spec = (tf.TensorSpec([1, img_size, img_size, 3], tf.float32, name=input_name),)
    from_saved_model(saved_model_dir, input_signature=spec, opset=13, output_path=onnx_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_model_artifact", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--target_device_name", required=True)
    parser.add_argument("--precision", choices=["fp16","int8"], default="fp16")
    parser.add_argument("--s3_calib_images", default=None)
    args = parser.parse_args()

    in_dir  = "/opt/ml/processing/input"
    out_dir = "/opt/ml/processing/output"
    work    = "/opt/ml/processing/work"

    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)

    # 1) Download and extract model.tar.gz
    tar_path = os.path.join(in_dir, "model.tar.gz")
    download_s3_to(tar_path, args.s3_model_artifact)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(work)

    # 2) Locate SavedModel
    sm_dir = find_saved_model_dir(work)
    if not sm_dir:
        print("ERROR: saved_model.pb not found in artifact", file=sys.stderr)
        sys.exit(2)

    # 3) Inspect signature
    sigs = list_signature(sm_dir)
    print("Detected signatures:\n", json.dumps(sigs, indent=2))

    # Require numeric float32 input; fail fast if DT_STRING
    sig = sigs.get("serving_default") or next(iter(sigs.values()))
    has_string_input = any(dt == "string" for dt, _ in sig["inputs"].values())
    if has_string_input:
        print("ERROR: serving_default has DT_STRING input (bytes). Re-export numeric input SavedModel first.", file=sys.stderr)
        sys.exit(3)

    # Identify input tensor name (first key) and convert to ONNX
    input_name = next(iter(sig["inputs"].keys()))
    onnx_path = os.path.join(out_dir, "model.onnx")
    convert_to_onnx(sm_dir, onnx_path, img_size=args.img_size, input_name=input_name)

    # 4) (Optional) Download calibration images
    calib_local = None
    if args.precision == "int8" and args.s3_calib_images:
        calib_local = os.path.join(in_dir, "calib_images")
        os.makedirs(calib_local, exist_ok=True)
        sync_s3_dir_to_local(args.s3_calib_images, calib_local)

    # 5) Compile with Qualcomm AI Hub
    import qai_hub as hub
    token = os.environ.get("QAIHUB_API_TOKEN")
    if not token:
        print("ERROR: QAIHUB_API_TOKEN not set in environment.", file=sys.stderr)
        sys.exit(4)

    hub.login(api_token=token)
    device = hub.Device(args.target_device_name)

    opts = f"--precision {args.precision}"
    job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        name="mobilenetv3_compile_job",
        input_specs={input_name: ((1, args.img_size, args.img_size, 3), "float32")},
        calibration_data=calib_local if args.precision == "int8" else None,
        options=opts,
    )
    print("Submitted compile job:", job.id)
    # Persist job metadata
    with open(os.path.join(out_dir, "compile_job.json"), "w") as f:
        json.dump({"job_id": job.id}, f)

if __name__ == "__main__":
    main()

------

import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# TensorFlow CPU image is enough (conversion only). Example for us-east-1 TF 2.12 CPU:
image_uri = sagemaker.image_uris.retrieve(
    framework="tensorflow",
    region=sess.boto_region_name,
    version="2.12",
    image_scope="training",   # training image has TF & pip
    instance_type="ml.m5.xlarge"
)

sp = ScriptProcessor(
    image_uri=image_uri,
    role=role,
    command=["python3"],
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name=JOB_NAME_PREFIX,
    env={"QAIHUB_API_TOKEN": AI_HUB_API_TOKEN},
)

# The script will pip-install its own deps (tf2onnx, qai-hub) inside the container command
sp.run(
    code="src/convert_and_compile.py",
    inputs=[],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{sess.default_bucket()}/qaihub/compile_outputs/")
    ],
    arguments=[
        "--s3_model_artifact", S3_MODEL_ARTIFACT,
        "--img_size", str(IMG_SIZE),
        "--target_device_name", TARGET_DEVICE_NAME,
        "--precision", PRECISION,
        "--s3_calib_images", S3_CALIB_IMAGES or "",
    ],
    container_entrypoint=["/bin/bash","-lc",
        # install minimal deps & run
        "pip install -q tf2onnx==1.15.1 qai-hub onnx && "
        "python3 /opt/ml/processing/input/code/convert_and_compile.py "
        "--s3_model_artifact \"$0\" --img_size \"$1\" --target_device_name \"$2\" --precision \"$3\" --s3_calib_images \"$4\"",
        S3_MODEL_ARTIFACT, str(IMG_SIZE), TARGET_DEVICE_NAME, PRECISION, S3_CALIB_IMAGES or ""
    ],
)
