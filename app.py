from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

app = Flask(__name__)

# Path to the folder where uploaded images will be stored
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your custom model
custom_model = load_model('D:\journal paper\project_folder\m1.h5')  # Replace with your model file path

# Define your classes
classes = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]  # Replace with your classes

# Preprocess image function for your custom model
def preprocess_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(120, 120))
        preprocessed_img = preprocess_image(img)
        
        # Make prediction using your custom model
        preds = custom_model.predict(preprocessed_img)
        predicted_class_index = np.argmax(preds)
        predicted_class = classes[predicted_class_index]
        
        return render_template('result.html', filename=filename, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
