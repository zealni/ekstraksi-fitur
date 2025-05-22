from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model dan fitur dataset
model = load_model('efficientnet_model.h5')
with open('features.pkl', 'rb') as f:
    features, filenames = pickle.load(f)

# Fungsi untuk ekstraksi fitur dari gambar input
def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array)
    return feature[0]

# Fungsi bantu untuk membersihkan nama file
def clean_title(filename):
    name = os.path.splitext(filename)[0]  # Hapus ekstensi
    name = ''.join([c for c in name if not c.isdigit()])  # Hapus angka
    name = name.replace('_', ' ').replace('-', ' ')  # Ganti _ dan - jadi spasi
    return name.strip().title()  # Capitalize

# Route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        upload_path = os.path.join('static', 'uploaded.jpg')
        file.save(upload_path)

        input_feature = extract_feature(upload_path)
        sims = cosine_similarity([input_feature], features)[0]
        top_indices = sims.argsort()[-5:][::-1]

        results = []
        for idx in top_indices:
            filename = os.path.basename(filenames[idx])
            title = clean_title(filename)

            results.append({
                'path': f'images/{filename}',  # aman untuk URL web
                'score': f"{sims[idx]*100:.2f}%",
                'title': title
            })

        return render_template('index.html', uploaded=True, image_path='uploaded.jpg', results=results)

    return render_template('index.html', uploaded=False)

# Jalankan Flask
if __name__ == '__main__':
    app.run(debug=True)
