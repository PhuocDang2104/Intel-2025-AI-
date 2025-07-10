# 🍍 Edge AIoT System for Fruit Crop Monitoring

A Flask-based Edge AIoT system using Intel NCS2 & OpenVINO for:
- ✅ Fruit quality & ripeness detection
- ✅ Counting fruit yield
- ✅ Threat detection: fire, intrusion, illegal logging

## 📂 Project Structure
- `web_flask/` — Flask server + UI
- `ai_models/` — AI model folders (YOLO, OpenVINO, etc)
- `static/`, `templates/` — Web UI
- `Dockerfile`, `docker-compose.yml` — Container setup
- `.github/workflows/ci.yml` — CI/CD build automation

## 🚀 Run Locally
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd web_flask
python app.py