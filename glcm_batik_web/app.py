import os
from PIL import Image
import io as IO
import numpy as np
from flask import Flask, request, render_template, redirect, send_from_directory, jsonify, url_for, send_file
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
    model = joblib.load('glcm_batik_web/batik_classifier_svm_rbf.joblib')
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

def noisy(noise_typ, image):
    image = image.astype(np.float32)  # ensure float for noise operations

    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 10
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        noisy = np.copy(image)

        # Salt
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        noisy[tuple(coords)] = 255

        # Pepper
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        noisy[tuple(coords)] = 0

    elif noise_typ == "poisson":
        noisy = np.random.poisson(image).astype(np.float32)

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        noisy = image + image * gauss

    else:
        raise ValueError("Unsupported noise type")

    # ✅ Convert to valid uint8 image
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def noise_filter(method, grayscale_image: np.ndarray):
    buf = grayscale_image.copy()

    if method == 'batas':
        for h in range(1, grayscale_image.shape[0]):  # Loop melalui tinggi gambar (dimulai dari 1)
            for w in range(1, grayscale_image.shape[1]):  # Loop melalui lebar gambar (dimulai dari 1)
                min_pixel = np.min(grayscale_image[h-1:h+1, w-1:w+1])  # Mencari nilai piksel minimum di sekitar
                max_pixel = np.max(grayscale_image[h-1:h+1, w-1:w+1])  # Mencari nilai piksel maksimum di sekitar
                if grayscale_image[h, w] < min_pixel:
                    buf[h, w] = min_pixel  # Jika piksel lebih kecil dari batas bawah, ubah ke min_pixel
                elif grayscale_image[h, w] > max_pixel:
                    buf[h, w] = max_pixel  # Jika piksel lebih besar dari batas atas, ubah ke max_pixel
        return buf  # Mengembalikan gambar hasil filter
    elif method == 'mean':
        for h in range(1, grayscale_image.shape[0] - 1):  # Loop untuk tinggi gambar (menghindari tepi)
            for w in range(1, grayscale_image.shape[1] - 1):  # Loop untuk lebar gambar (menghindari tepi)
                # Menghitung rata-rata nilai piksel dari jendela 3x3 di sekitar piksel (h, w)
                buf[h, w] = np.mean(grayscale_image[h-1:h+2, w-1:w+2])
        return buf  # Mengembalikan gambar hasil filter
    elif method == 'median':
        for h in range(1, grayscale_image.shape[0] - 1):  # Loop untuk tinggi gambar (menghindari tepi)
            for w in range(1, grayscale_image.shape[1] - 1):  # Loop untuk lebar gambar (menghindari tepi)
                # Menghitung median dari jendela 3x3 di sekitar piksel (h, w)
                buf[h, w] = np.median(grayscale_image[h-1:h+2, w-1:w+2])
        return buf  # Mengembalikan gambar hasil filter median
    else:
        raise ValueError("Metode filter tidak dikenali: harus 'batas', 'mean', atau 'median'")

# Rute Halaman
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({ "message": "Invalid Input" }), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({ "message": "Invalid Input" }), 400
        if file and allowed_file(file.filename) and model:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            features = extract_features(filepath)
            prediction = model.predict([features])

            return jsonify({ "filename": f"/uploads/{filename}", "prediction": prediction[0] })
        else:
            return jsonify({ "message": "Invalid Input" }), 400

    # Render halaman utama (GET request)
    return render_template('index.html', filename=None, prediction=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.realpath(app.config['UPLOAD_FOLDER']), filename)

@app.post('/filter-noise')
def filter_noise():
    if request.form['method'] not in ["batas", "mean", "median"]:
        return jsonify({ "message": "Invalid Input" }), 400
    method_filter = request.form['method']
    if 'file' not in request.files:
        return jsonify({ "message": "Invalid Input" }), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({ "message": "Invalid Input" }), 400
    if file and allowed_file(file.filename):
        image = Image.open(file).convert('L')

        img_array = np.array(image)

        denoised_img = noise_filter(method_filter, img_array)

        result_image = Image.fromarray(denoised_img)
        img_io = IO.BytesIO()
        result_image.save(img_io, "PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")
    else:
        return jsonify({ "message": "Error Process" }), 400

@app.route('/add-noise', methods=['GET', 'POST'])
def add_noise():
    if request.method == 'POST':
        if request.form['type'] not in ["gauss", "s&p", "poisson", "speckle"]:
            return jsonify({ "message": "Invalid Input" }), 400
        noise_type = request.form['type']
        if 'file' not in request.files:
            return jsonify({ "message": "Invalid Input" }), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({ "message": "Invalid Input" }), 400
        if file and allowed_file(file.filename) and model:
            image = Image.open(file).convert('RGB')

            img_array = np.array(image)

            noisy_img = noisy(noise_type, img_array)

            if isinstance(noisy_img, np.ndarray):
                result_image = Image.fromarray(noisy_img)
                img_io = IO.BytesIO()
                result_image.save(img_io, "PNG")
                img_io.seek(0)

                return send_file(img_io, mimetype="image/png")
            else:
                return jsonify({ "message": "Error Process" }), 400
        else:
            return jsonify({ "message": "Error Process" }), 400

    return render_template('noise.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)