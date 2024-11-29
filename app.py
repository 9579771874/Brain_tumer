from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(_name_)

# Load the model once at startup
model = tf.keras.models.load_model('effnet.keras')  # Ensure the model path is correct

# Predefined labels
LABELS = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is included in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Open and preprocess the image
        img = Image.open(file).convert('RGB')  # Convert to RGB to ensure compatibility
        img = img.resize((150, 150))  # Resize to match model input
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction
        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction, axis=1)[0])
        result = LABELS[class_index]

        return jsonify({"prediction": result}), 200

    except Exception as e:
        # Handle any errors gracefully
        return jsonify({"error": str(e)}), 500
