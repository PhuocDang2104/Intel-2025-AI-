# 🍍 AIMING - AIoT Infravision for Monitoring, Inspection & Grading in Agriculture

An Edge AIoT system using Intel Embedded Computer BOXNUC & OpenVINO for:
- ✅ Recognize fruit type, count quantity, detect ripeness, and identify external defects. ( AI Computer Vision)
- ✅ Measure °Brix (glucose level), moisture content, detect internal bruises, and identify pest or fungal infections. ( AI NIR Spectral Analysis)



## 📂 Project Structure
- `web_flask/` — Flask server + UI
- `ai_models/` — AI model folders (Tensorflow, OpenVINO, etc)
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