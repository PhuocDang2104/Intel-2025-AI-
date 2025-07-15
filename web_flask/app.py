from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import supervision as sv
from openvino.runtime import Core
import time
import joblib
# ⚙️ Load scaler
scaler = joblib.load('../ai_models/AI Spectral MLP/scaler.pkl')

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, async_mode='threading')



cam =cv2.VideoCapture(0)
if cam.isOpened():
    print("✅ Camera initialized successfully")
else:
    time.sleep(1)

core = Core()
device = 'GPU'

det_model = core.read_model('../ai_models/Detection AI/best.xml')
det_compiled = core.compile_model(det_model, device)
det_input_name = det_compiled.input(0).any_name

apple_model = core.read_model("../ai_models/Detection AI/apple_ripeness.xml")
apple_compiled = core.compile_model(apple_model, device)
apple_input_name = apple_compiled.input(0).any_name

mango_model = core.read_model("../ai_models/Detection AI/model.xml")
mango_compiled = core.compile_model(mango_model, device)
mango_input_name = mango_compiled.input(0).any_name

quality_model = core.read_model("../ai_models/AI Spectral MLP/openvino_model/nir_model_v3.xml")
quality_compiled = core.compile_model(quality_model, device)
quality_input_name = quality_compiled.input(0).any_name

class_names = ["Apple", "Dau_chin", "Dau_chua_chin", "Mango"]
ripeness_label_mango = ["7", "7", "9", "10"]
ripeness_label_apple = ["10", "5", "7", "9"]

spectral_profiles = {
    'apple_0':   [0.409, 0.395, 0.591, 0.694, 0.939, 1.146],
    'apple_1':   [0.393, 0.393, 0.605, 0.667, 0.917, 1.129],
    'apple_2':   [0.394, 0.392, 0.602, 0.666, 0.9175, 1.1293],
    'apple_3':   [0.4095, 0.3952, 0.5911, 0.694, 0.9393, 1.1456],
    'mango_0':   [0.384, 0.396, 0.657, 0.702, 0.806, 0.9291],
    'mango_1':   [0.383, 0.395, 0.656, 0.7027, 0.8064, 0.92751],
    'mango_2':   [0.384, 0.396, 0.657, 0.702, 0.806, 0.9291],
    'mango_3':   [0.384, 0.3963, 0.657, 0.7302, 0.806, 0.92951],
}

latest_result = {
    'fruit_type': '',
    'fruit_type_confidence': 0.0,
    'ripeness': 0.0,
    'ripeness_confidence': 0.0,
    'state': 'waiting',
    'quality_prediction': []
}

def classify_fruit(crop, model, input_name):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    norm = img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(norm, axis=0)
    result = model.infer_new_request({input_name: input_tensor})
    output = list(result.values())[0]
    ripeness = int(np.argmax(output))
    confidence = float(np.max(output))
    return ripeness, confidence

def infer_quality(fruit, ripeness):
    key = f"{fruit.lower()}_{ripeness}"
    spectral = spectral_profiles.get(key)
    if spectral is None:
        return [0.0] * 6, "N/A", 0.0

    # ✅ Ánh xạ ripeness index về giá trị thực (5–10)
    ripeness_value = {
        'apple': [10, 5, 7, 9],
        'mango': [5, 7, 9, 10]
    }.get(fruit.lower(), [0, 0, 0, 0])[ripeness]

    print(f"🧪 Ripeness mapped: index {ripeness} → value {ripeness_value}")

    fruit_one_hot = {
        'apple': [1, 0, 0],
        'mango': [0, 1, 0],
        'strawberry': [0, 0, 1],
    }.get(fruit.lower(), [0, 0, 0])

    input_vector = np.array([spectral + [ripeness_value] + fruit_one_hot], dtype=np.float32)
    input_scaled = scaler.transform(input_vector)
    input_scaled = input_scaled.reshape((1, 10, 1))

    print("🧪 Fruit:", fruit)
    print("🧪 One-hot vector:", fruit_one_hot)
    print("🧪 Final input vector:", input_vector)
    result = quality_compiled.infer_new_request({quality_input_name: input_scaled})
    outputs = {out.get_any_name(): out for out in quality_compiled.outputs}

    pred_reg    = result[outputs['regression']]
    pred_grade  = result[outputs['grade']]
    pred_defect = result[outputs['defect']]
    pred_fungus = result[outputs['fungus']]
    print("🔍 Regression (°Brix, Moisture):", pred_reg)
    print("🔍 Grade class logits (softmax):", pred_grade)
    print("🔍 Defect probability:", pred_defect)
    print("🔍 Fungus/Disease probability:", pred_fungus)
    # 🔍 Lấy grade label từ softmax
    grade_index = int(np.argmax(pred_grade[0]))
    grade_label = ['A', 'B', 'C'][grade_index]

    grade_conf = float(np.max(pred_grade[0]))
    

    grade_conf = float(np.max(pred_grade[0]))
    prediction = [
        float(pred_reg[0][0]),      # Brix
        float(pred_reg[0][1]),      # Moisture
        grade_conf,                 # Grade confidence (0–1)
        float(pred_defect[0][0]),   # Defect probability (0–1)
        float(pred_fungus[0][0])    # Fungus probability (0–1)
    ]
    
    return prediction, grade_label, ripeness_value
    
print("🧠 Model Output Names:")
for i, out in enumerate(quality_compiled.outputs):
    print(f"{i}: {out.get_any_name()}")

def process_frames():
    global latest_result
    while True:
        success, frame = cam.read()
        if not success:
            break

        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (320, 320))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]

        result = det_compiled.infer_new_request({det_input_name: input_tensor})
        output = list(result.values())[0][0]
        boxes = output[:4, :].T
        probs = output[4:, :].T
        scores = np.max(probs, axis=1)
        classes = np.argmax(probs, axis=1)
        mask = scores > 0.8
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

        if len(boxes) > 0:
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes[:, [0, 2]] *= w / 320
            boxes[:, [1, 3]] *= h / 320
            boxes = boxes.astype(int)

            detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=classes)
            labels = []
            latest_result['state'] = 'running'

            for i, (box, cls_id) in enumerate(zip(boxes, classes)):
                x1, y1, x2, y2 = box
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                else:
                    continue

                if cls_id == 0:
                    fruit = 'apple'
                    ripeness, conf = classify_fruit(crop, apple_compiled, apple_input_name)
                elif cls_id == 3:
                    fruit = 'mango'
                    ripeness, conf = classify_fruit(crop, mango_compiled, mango_input_name)
                else:
                    continue

                # ✅ Chính xác:
                quality, grade_label, ripeness_value= infer_quality(fruit, ripeness)

                latest_result.update({
                    'fruit_type': fruit,
                    'fruit_type_confidence': float(scores[i] * 100),
                    'ripeness': ripeness_value,
                    'ripeness_confidence': float(conf * 100),
                    'state': 'done',
                    'quality_prediction': quality,
                    'grade_class': grade_label
                })

                ripeness_label = ripeness_label_apple if fruit == 'apple' else ripeness_label_mango
                label = f"{fruit.capitalize()} R:{ripeness_label[ripeness]} Q:{[round(float(q), 2) for q in quality]}"
                labels.append(label)

            socketio.emit('update_detection', latest_result)
            box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        else:
            latest_result.update({
                'fruit_type': '',
                'fruit_type_confidence': 0.0,
                'ripeness': 0.0,
                'ripeness_confidence': 0.0,
                'state': 'waiting',
                'quality_prediction': [],
            })
            socketio.emit('update_detection', latest_result)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/home')
def main_system():
    return render_template('interfere.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    emit('update_detection', latest_result)
    print("Client connected, sent initial data:", latest_result)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
