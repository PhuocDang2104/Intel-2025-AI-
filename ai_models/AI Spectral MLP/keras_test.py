import numpy as np
import joblib
from keras.layers import TFSMLayer
import tensorflow as tf

# ⚙️ Load scaler đã lưu
scaler = joblib.load(r"C:\Users\ADMIN\Desktop\Intel 2025 AI\ai_models\AI Spectral MLP\scaler.pkl")

# ⚙️ Load mô hình từ thư mục SavedModel
model = TFSMLayer(
    r"C:\Users\ADMIN\Desktop\Intel 2025 AI\ai_models\AI Spectral MLP\nir_model_v3_savedmodel",
    call_endpoint="serving_default"
)

# 📥 Input thủ công
manual_input = np.array([[  
    0.3945, 0.4124, 0.5325,     # 610nm, 680nm, 730nm
    0.6438, 0.9651, 1.1773,     # 760nm, 810nm, 860nm
    9,                          # ripeness
    0, 0, 1                     # one-hot fruit: strawberry
]], dtype=np.float32)

# 🔁 Chuẩn hóa và reshape
manual_input_scaled = scaler.transform(manual_input).reshape((1, 10, 1))

# 🚀 Inference
outputs = model(manual_input_scaled)

# 🧾 Phân tách các output
pred_reg = outputs["output_0"].numpy()
pred_grade = outputs["output_1"].numpy()
pred_defect = outputs["output_2"].numpy()
pred_fungus = outputs["output_3"].numpy()

# 📊 Hiển thị
grade_map = {0: 'A', 1: 'B', 2: 'C'}
print("🎯 Inference Result (SavedModel)")
print("────────────────────────────────────────────")
print(f"📌 Predicted Brix (°Bx):      {pred_reg[0][0]:.2f}")
print(f"📌 Predicted Moisture (%):    {pred_reg[0][1]:.2f}")
print()
print(f"🏷️ Grade Prediction:          {grade_map[np.argmax(pred_grade[0])]} → Confidence: {np.max(pred_grade[0]):.2%}")
print(f"⚠️ Internal Defect:           {'Yes' if pred_defect[0][0] > 0.5 else 'No'} → Confidence: {pred_defect[0][0]:.2%}")
print(f"🍄 Fungus Infection:          {'Yes' if pred_fungus[0][0] > 0.5 else 'No'} → Confidence: {pred_fungus[0][0]:.2%}")
