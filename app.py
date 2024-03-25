from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read and preprocess the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (540, 420))  # Resize to match model input shape
    image = image.reshape((1, 420, 540, 1)) / 255.0  # Normalize

    # Predict denoised image
    denoised_image = model.predict(image)
    denoised_image = (denoised_image[0, :, :, 0] * 255).astype(np.uint8)

    # Encode denoised image as base64
    _, buffer = cv2.imencode('.jpg', denoised_image)
    denoised_image_b64 = buffer.tobytes()

    return jsonify({'denoised_image': denoised_image_b64.decode('utf-8')})

if __name__ == '__main__':
    app.run(debug=True)
