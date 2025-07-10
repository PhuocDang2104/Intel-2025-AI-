from flask import Flask, render_template_string
import pandas as pd

app = Flask(__name__)

@app.route('/')
def show_table():
    file_path = r'C:\Users\ADMIN\Desktop\Intel 2025 AI\ai_models\AI Spectral MLP\data_nir\nir_spectral_dataset_augmented.csv'
    delimiters = [',', ';', '\t']

    for d in delimiters:
        try:
            df = pd.read_csv(file_path, delimiter=d, encoding='utf-8')
            if df.shape[1] > 1:
                break  # thành công
        except Exception:
            continue
    else:
        return "<h2>Không đọc được CSV - hãy kiểm tra file hoặc gửi mình để xử lý.</h2>"

    # Hiển thị bảng HTML
    table_html = df.to_html(classes='table table-bordered table-striped', index=False)
    return render_template_string("""
    <!doctype html>
    <html>
    <head>
        <title>CSV Viewer</title>
        <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="p-4">
        <h1 class="mb-4">Dữ liệu trái cây theo phổ NIR và Brix, ripeness, grading</h1>
        {{ table|safe }}
    </body>
    </html>
    """, table=table_html)

if __name__ == '__main__':
    app.run(debug=True)
