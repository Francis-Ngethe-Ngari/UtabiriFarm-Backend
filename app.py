# Import necessary dependencies
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import datetime

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = '/home/sudotechpro/alx_learn/UtabiriFarm-Backend/models/utabirifarm-model.keras'
model = tf.keras.models.load_model(model_path)

# Provide a list of class names corresponding to your model's output classes.
class_names = [
    # Maize classes
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot(2)',
    'Corn_(maize)___Common_rust_(2)',
    'Corn_(maize)___Northern_Leaf_Blight(2)',
    'Corn_(maize)___healthy(2)',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot(3)',
    'Corn_(maize)___Common_rust_(3)',
    'Corn_(maize)___Northern_Leaf_Blight(3)',
    'Corn_(maize)___healthy(3)',
    # Tomato classes
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
    'Tomato___Bacterial_spot(2)',
    'Tomato___Early_blight(2)',
    'Tomato___Late_blight(2)',
    'Tomato___Leaf_Mold(2)',
    'Tomato___Septoria_leaf_spot(2)',
    'Tomato___Spider_mites Two-spotted_spider_mite(2)',
    'Tomato___Target_Spot(2)',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus(2)',
    'Tomato___Tomato_mosaic_virus(2)',
    'Tomato___healthy(2)',
    'Tomato___Bacterial_spot(3)',
    'Tomato___Early_blight(3)',
    'Tomato___Late_blight(3)',
    'Tomato___Leaf_Mold(3)',
    'Tomato___Septoria_leaf_spot(3)',
    'Tomato___Spider_mites Two-spotted_spider_mite(3)',
    'Tomato___Target_Spot(3)',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus(3)',
    'Tomato___Tomato_mosaic_virus(3)',
    'Tomato___healthy(3)',
    # Potato classes
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Potato___Early_blight(2)',
    'Potato___Late_blight(2)',
    'Potato___healthy(2)',
    'Potato___Early_blight(3)',
    'Potato___Late_blight(3)',
    'Potato___healthy(3)'
    ]

# Allowed Extentions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create /prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Make prediction
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        predicted_class = class_names[np.argmax(predictions)]

        # # Disease information
        # disease_info = {
        #     'Class1': {'symptoms': '...', 'treatment': '...'},
        #     # Add more classes
        # }
        # info = disease_info.get(predicted_class, {})

        # Path to predictions dir
        predict_path = '/home/sudotechpro/alx_learn/UtabiriFarm-Backend/predictions'


        def save_prediction(user_id, image_filename, predicted_class, confidence):
            # Create a directory for storing predictions if it doesn't exist
            os.makedirs(predict_path, exist_ok=True)
            
            # Define the filename (e.g., predictions.txt)
            prediction_file = os.path.join(predict_path, 'predictions.txt')
            
            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save prediction details
            with open(prediction_file, 'a') as f:
                f.write(f'{timestamp}, User ID: {user_id}, Image: {image_filename}, '
                        f'Prediction: {predicted_class}, Confidence: {confidence:.4f}\n')


        # Save prediction
        user_id = 'test-user'  # Replace with actual user ID if available
        save_prediction(user_id, file.filename, predicted_class, confidence)

        # Build response
        response = {
            'class': predicted_class,
            'confidence': confidence,
            # 'info': info
        }

        return jsonify(response)
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400
    
    
if __name__ == '__main__':
    app.run(debug=True)





