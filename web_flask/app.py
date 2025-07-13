
from flask import Flask, render_template, jsonify

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def home():
    return render_template('login.html')


@app.route('/home')
def main_system():
    return render_template('interfere.html',
        fruit_type="Mango",
        fruit_type_confidence=93.2,
        ripeness=7.8,
        ripeness_confidence=88.6,
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)