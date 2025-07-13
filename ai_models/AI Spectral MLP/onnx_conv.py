import tensorflow as tf
import tf2onnx

# ✅ Load lại model mà không cần compile
model = tf.keras.models.load_model("nir_model_v3.keras", compile=False)

# ✅ Xác định input shape
spec = (tf.TensorSpec((None, 10, 1), tf.float32, name="input"),)

# ✅ Chuyển sang ONNX
output_path = "nir_model_v3.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print("✅ Converted to ONNX:", output_path)