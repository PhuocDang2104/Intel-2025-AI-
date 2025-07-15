import joblib
import numpy as np
from openvino.runtime import Core

# ⚙️ Load scaler đã lưu từ lúc training
scaler = joblib.load(r"C:\Users\ADMIN\Desktop\Intel 2025 AI\ai_models\AI Spectral MLP\scaler.pkl")

# ⚙️ Load OpenVINO model
ie = Core()
model = ie.read_model("nir_model_v3.xml")
compiled_model = ie.compile_model(model, device_name="CPU")

# 🧾 Input features theo đúng thứ tự huấn luyện
manual_input = np.array([[
    0.39, 0.40, 0.49
,  # NIR wavelengths: 610, 680, 730
    0.54
, 0.8606887301626904
, 0.8903615507678188
,  # NIR wavelengths: 760, 810, 860
    9,                      # ripeness
    1, 0, 0                 # one-hot fruit
]], dtype=np.float32)

# 🔁 Chuẩn hóa bằng scaler đã huấn luyện
manual_input_scaled = scaler.transform(manual_input)

# ➕ Reshape lại đúng input shape của model
manual_input_scaled = manual_input_scaled.reshape((1, 10, 1))

# 🚀 Inference
results = compiled_model([manual_input_scaled])

# 🧾 Xử lý output
pred_reg = results[compiled_model.output(0)]
pred_grade = results[compiled_model.output(1)]
pred_defect = results[compiled_model.output(2)]
pred_fungus = results[compiled_model.output(3)]

# 🖨️ Hiển thị kết quả
grade_map = {0: 'A', 1: 'B', 2: 'C'}
print("🎯 Inference Result (OpenVINO)")
print(f"📌 Predicted Brix (°Bx):      {pred_reg[0][0]:.2f}")
print(f"📌 Predicted Moisture (%):    {pred_reg[0][1]:.2f}")
print()
print(f"🏷️ Grade Prediction:          {grade_map[np.argmax(pred_grade[0])]} → Confidence: {np.max(pred_grade[0]):.2%}")
print(f"⚠️ Internal Defect:           {'Yes' if pred_defect[0][0] > 0.5 else 'No'} → Confidence: {pred_defect[0][0]:.2%}")
print(f"🍄 Fungus Infection:          {'Yes' if pred_fungus[0][0] > 0.5 else 'No'} → Confidence: {pred_fungus[0][0]:.2%}")

