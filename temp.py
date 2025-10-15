!pip install --upgrade pip setuptools wheel
!pip install --no-cache-dir tensorflow-macos==2.13 tensorflow-metal==1.1.0


import tensorflow as tf
print(tf.__version__)


!python -m tensorflow.python.tools.saved_model_cli show \
  --dir "/path/to/saved_model" \
  --tag_set serve \
  --signature_def serving_default
