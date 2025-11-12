from sagemaker.predictor import Predictor
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import IdentitySerializer

pretrained_predictor = Predictor(endpoint_name="your-endpoint-name")
pretrained_predictor.serializer = IdentitySerializer("application/x-image")
pretrained_predictor.deserializer = JSONDeserializer()

result = pretrained_predictor.predict(img_bytes)
