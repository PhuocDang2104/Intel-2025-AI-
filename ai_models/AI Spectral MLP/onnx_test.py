import onnxruntime as ort
import numpy as np

# ⚙️ Load ONNX model
session = ort.InferenceSession("nir_model_v3.onnx")

# 🔍 Lấy tên input layer
input_name = session.get_inputs()[0].name

# 🧾 In tên các output layer
output_names = [output.name for output in session.get_outputs()]
print("📌 Output names:", output_names)

# 📥 Tạo input mẫu (float32, đúng shape và scaler nếu cần)
manual_input = np.array([[ 
    0.3945, 0.4123584955046491, 0.5325196501093834,   # 610, 680, 730
    0.6438129337936717, 0.9651213672824536, 1.1773568011848727,  # 760, 810, 860
    9, 0, 0, 1  # ripeness + one-hot strawberry
]], dtype=np.float32)

# ⚠️ Nhớ reshape đúng (giống như model Keras khi training)
manual_input = manual_input.reshape((1, 10, 1))

# 🚀 Inference
results = session.run(output_names, {input_name: manual_input})

# 🖨️ Hiển thị kết quả
print("🎯 Inference Result (ONNX)")
print("────────────────────────────────────────────")
print(f"📌 Predicted Brix (°Bx):      {results[0][0][0]:.2f}")
print(f"📌 Predicted Moisture (%):    {results[0][0][1]:.2f}")

grade_map = {0: 'A', 1: 'B', 2: 'C'}
grade_pred = results[1][0]
print(f"🏷️ Grade Prediction:          {grade_map[np.argmax(grade_pred)]} → Confidence: {np.max(grade_pred):.2%}")

defect_pred = results[2][0][0]
fungus_pred = results[3][0][0]
print(f"⚠️ Internal Defect:           {'Yes' if defect_pred > 0.5 else 'No'} → Confidence: {defect_pred:.2%}")
print(f"🍄 Fungus Infection:          {'Yes' if fungus_pred > 0.5 else 'No'} → Confidence: {fungus_pred:.2%}")
