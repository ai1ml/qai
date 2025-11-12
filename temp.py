from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

# same model_id/model_version you fine-tune later
baseline_model = JumpStartModel(model_id=model_id, model_version=model_version)
baseline_predictor = baseline_model.deploy(
    initial_instance_count=1,
    instance_type=endpoint_instance_type,
    endpoint_name="baseline-pretrained-endpoint",
)

# match your existing predictor I/O
baseline_predictor.serializer   = IdentitySerializer(content_type="application/x-image")
baseline_predictor.deserializer = JSONDeserializer()
baseline_predictor.accept       = "application/json;verbose"

---------

print("=== Baseline (pretrained) validation ===")
predictor = baseline_predictor   # <-- one line swap
# ➜ now run your existing validation/inference block as-is
# (after it finishes, capture any metrics you print, e.g.)
acc_baseline = acc
cm_baseline  = cm
