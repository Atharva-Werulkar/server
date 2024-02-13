
from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
# from tensorflow.keras.models import load_model
# from pydub import AudioSegment

# Load the trained model
xgb = joblib.load('xgboost_model.pkl')

# Function to preprocess the audio file
def extract_features(audio_files):
    features = []
    for file in audio_files:
        # Load audio file
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        # Extract audio features (example: MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Calculate mean of MFCCs across time
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features.append(mfccs_mean)
        print(np.array(features))  # Print feature array for debugging
    return np.array(features)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    try:
        # Check if the POST request contains the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        # Get the file from the POST request
        audio_file = request.files['file']
        
        # Preprocess the audio file
        processed_audio_file = extract_features(audio_file)
        
        # Perform prediction using the loaded model
        prediction = xgb.predict(processed_audio_file)
        # Note: You need to adapt this part to fit your model's input requirements
        
        # Dummy prediction for testing
       # prediction = 1
        
        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
