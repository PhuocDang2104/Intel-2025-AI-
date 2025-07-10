FROM python:3.10-slim

# Cài thêm thư viện hệ thống nếu cần OpenCV, v.v
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ app (web_flask) vào container
COPY web_flask/ /app

# Chạy Flask app
CMD ["python", "app.py"]
