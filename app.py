from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os
import os

# model_path = os.path.join(os.getcwd(), 'model', 'effnet.keras')
model = load_model('effnet.keras')

# Flask app setup
app = Flask(__name__)
app.secret_key = 'Project_88'  # Required for session management

# # Load your pre-trained model
# model = load_model(os.path.join('model', 'final_model.keras'))

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Here you would add authentication logic
        if email == "user@example.com" and password == "password":  # Dummy validation
            session['user'] = email
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Here you would add logic to save the user in the database
        flash('Registration successful!')
        return redirect(url_for('login'))
    return render_template('register.html')

# Dashboard route (after login)
@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', user=session['user'])
    else:
        return redirect(url_for('login'))

# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# Tumor Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if an image is uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        # Process the image for prediction
        img = Image.open(file)
        img = img.resize((150, 150))
        img_array = np.array(img)
        img_array = img_array.reshape(1, 150, 150, 3) / 255.0  # Normalize the image
        # Predict using the model
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        labels = ['Glioma Tumor', 'No tumor', 'Meningioma Tumor', 'Pituitary Tumor']
        result = labels[class_index]
        return render_template('result.html', prediction=result)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)


