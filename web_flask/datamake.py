import pandas as pd
import numpy as np

# 📥 Bước 1: Đọc file gốc
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\Intel 2025 AI\ai_models\AI Spectral MLP\data_nir\NIR_fruit_dataset_final_v9.csv")

# 🧲 Bước 2: Lọc các mẫu mango grade A sẵn có để làm mẫu gốc
base = df[
    (df["fruit"].str.lower() == "mango") &
    (df["grade"] == "A") &
    (df["fungus"] == 0) &
    (df["defect"] == 0) &
    (df["ripeness"] > 8)
]

# Nếu không đủ mẫu, lấy thêm mango bất kỳ để làm base mẫu
if len(base) < 10:
    base = df[df["fruit"].str.lower() == "mango"].sample(10, random_state=42)

# 🔁 Bước 3: Tạo 200 sample mới
new_samples = []
for _ in range(200):
    row = base.sample(1).copy().iloc[0]

    # Thêm nhiễu nhẹ cho các cột số (trừ fungus, defect)
    for col in df.columns:
        if df[col].dtype != 'object' and col not in ['fungus', 'defect']:
            noise = np.random.normal(0, 0.5)
            row[col] = round(row[col] + noise, 2)

    # Đảm bảo điều kiện grade A
    row['fruit'] = 'mango'
    row['grade'] = 'A'
    row['fungus'] = 0
    row['defect'] = 0
    row['ripeness'] = max(8.1, row['ripeness'])  # chắc chắn >8

    new_samples.append(row)

# 🧩 Bước 4: Nối vào DataFrame gốc
df_augmented = pd.concat([df, pd.DataFrame(new_samples)], ignore_index=True)

# 💾 Bước 5: Lưu file mới
df_augmented.to_csv("NIR_fruit_dataset_final_v10.csv", index=False)

print("✅ Đã tạo file mới với 200 mẫu mango grade A.")
