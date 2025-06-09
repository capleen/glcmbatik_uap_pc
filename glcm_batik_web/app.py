import os
import numpy as np
from flask import Flask, request, render_template, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import joblib
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops

# Konfigurasi Awal
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memuat Model
try:
    model = joblib.load('batik_classifier_svm_rbf.joblib')
    print("Model SVM berhasil dimuat.")
except FileNotFoundError:
    print("Error: File model SVM tidak ditemukan. Pastikan Anda sudah melatih dan menyimpan model.")
    model = None

QUANT_LEVELS = 16
DISTANCE = [1]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

def quantize_image(img_gray, levels=16):
    return (img_gray * (levels - 1)).astype(np.uint8)

def glcm_entropy(glcm):
    p = glcm.astype(np.float64)
    p_sum = p.sum(axis=(0,1), keepdims=True)
    p_norm = p / (p_sum + 1e-10)
    return -np.sum(p_norm * np.log(p_norm + 1e-10), axis=(0,1))

def extract_features(image_path):
    img = io.imread(image_path)
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img.astype(np.float64) / 255.0
    img_q = quantize_image(img_gray, levels=QUANT_LEVELS)
    glcm = graycomatrix(img_q, distances=DISTANCE, angles=ANGLES, levels=QUANT_LEVELS, symmetric=True, normed=True)
    feats = []
    feats.extend(graycoprops(glcm, prop='energy').flatten())
    feats.extend(graycoprops(glcm, prop='contrast').flatten())
    feats.extend(graycoprops(glcm, prop='correlation').flatten())
    entropy_vals = glcm_entropy(glcm)
    feats.extend(entropy_vals.flatten())
    return feats

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rute Halaman
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename) and model:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            features = extract_features(filepath)
            prediction = model.predict([features])

            return render_template('index.html', filename=filename, prediction=prediction[0])

    # Render halaman utama (GET request)
    return render_template('index.html', filename=None, prediction=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)