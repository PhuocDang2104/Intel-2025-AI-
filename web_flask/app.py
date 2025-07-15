from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import supervision as sv
from openvino.runtime import Core
import threading

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, async_mode='threading')  # Use threading mode for compatibility

cam = cv2.VideoCapture(1)

core = Core()
device = 'GPU'

det_model = core.read_model('../ai_models/Detection AI/best.xml')
det_compiled = core.compile_model(det_model, device)
det_input_name = det_compiled.input(0).any_name

apple_model = core.read_model("../ai_models/Detection AI/apple_ripeness.xml")
apple_compiled = core.compile_model(apple_model, device)
apple_input_name = apple_compiled.input(0).any_name

mango_model = core.read_model("../ai_models/Detection AI/mango_ripeness.xml")
mango_compiled = core.compile_model(mango_model, device)
mango_input_name = mango_compiled.input(0).any_name
class_names = ["Apple", "Dau_chin", "Dau_chua_chin", "Mango"]
ripeness_label_mango = ["10", "9", "7", "5"]
ripeness_label_apple = ["10", "9", "7", "5"]

latest_result = {
    'fruit_type': '',
    'fruit_type_confidence': 0.0,
    'ripeness': 0.0,
    'ripeness_confidence': 0.0,
    'state': 'waiting'
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

def process_frames():
    global latest_result
    while True:
        success, frame = cam.read()
        if not success:
            print("Camera read failed")
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
        mask = scores > 0.78
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

        if len(boxes) > 0:
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes[:, [0, 2]] *= w / 320
            boxes[:, [1, 3]] *= h / 320
            boxes = boxes.astype(int)

            detections = sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=classes
            )

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
                    ripeness, conf = classify_fruit(crop, apple_compiled, apple_input_name)
                    ripeness_level = ripeness_label_apple[ripeness]
                elif cls_id == 3:
                    ripeness, conf = classify_fruit(crop, mango_compiled, mango_input_name)
                    ripeness_level = ripeness_label_mango[ripeness]
                else:
                    continue
                latest_result['fruit_type'] = class_names[cls_id]
                latest_result['fruit_type_confidence'] = float(scores[i] * 100)
                latest_result['ripeness'] = int(ripeness_level)
                latest_result['ripeness_confidence'] = float(conf * 100)
                latest_result['state'] = 'done'
                label = f"{class_names[cls_id]} R:{ripeness_level} ({conf:.2f})"
                labels.append(label)

            # Emit updated results
            socketio.emit('update_detection', latest_result)
            print("Emitted:", latest_result)  # Debug log

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
                'state': 'waiting'
            })
            socketio.emit('update_detection', latest_result)
            print("Emitted (no detection):", latest_result)  # Debug log

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
    return render_template('interfere.html',
        fruit_type=latest_result['fruit_type'],
        fruit_type_confidence=latest_result['fruit_type_confidence'],
        ripeness=latest_result['ripeness'],
        ripeness_confidence=latest_result['ripeness_confidence'],
        spectral_values="[321, 542, 432, 612, 451]",
        brix=12.3,
        brix_confidence=95.4,
        moisture=84.5,
        moisture_confidence=90.2,
        grade="A",
        grade_confidence=98.7,
        internal_defect_nir="No",
        internal_defect_confidence=91.3,
        disease_or_fungal="No",
        disease_confidence=87.5,
        camera_url="https://via.placeholder.com/640x360?text=Live+Camera+Feed",
        state="done"
    )

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    emit('update_detection', latest_result)
    print("Client connected, sent initial data:", latest_result)

# Start video processing in a background task
def start_video_processing():
    socketio.start_background_task(process_frames)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)