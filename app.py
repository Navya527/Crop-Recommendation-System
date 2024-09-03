from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', title="Crop Recommendation System")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = tf.keras.models.load_model('crop.h5')

# Load the label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

# Load the standard scaler
        with open('standard_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

# Get user input from command line arguments
    
        features = [float(x) for x in request.form.values()]
        features_scaled = scaler.transform([features])
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        user_input_df = pd.DataFrame([feature_names], columns=feature_names)
        user_input_scaled = scaler.transform(user_input_df)
# Predict the crop
        prediction = model.predict(user_input_scaled)

# Top 3 Recommendations
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_crops = label_encoder.inverse_transform(top_3_indices)
        top_3_probs = prediction[0][top_3_indices]


        recommendations = {crop: prob for crop, prob in zip(top_3_crops, top_3_probs)}
        return render_template('result.html', recommendations=recommendations, title="Prediction Result")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True,threaded=True)
