from flask import Flask, request, jsonify
import joblib
import numpy as np
import librosa
import os

# Load the XGBoost model
xgb = joblib.load('xgboost_model.pkl')

app = Flask(__name__)

def extract_features(audio_files):
    features = []
    for file in audio_files:
        try:
            # Load audio file
            audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
            # Extract audio features (example: MFCCs)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            # Calculate mean of MFCCs across time
            mfccs_mean = np.mean(mfccs.T, axis=0)
            features.append(mfccs_mean.tolist())  # Convert to list for JSON serialization
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    return np.array(features)

@app.route('/', methods=['POST'])

def predict():
    try:
        # Get audio file from request
        file = request.files['file']
        # Save the audio file temporarily
        audio_path = 'temp_audio.wav'  
        file.save(audio_path)
        
        # Extract features from the audio file
        features = extract_features([audio_path])

        # Perform prediction
        prediction = xgb.predict(features)

        # Convert prediction to list for JSON serialization
        prediction_list = prediction.tolist()

        # Return the predictions as JSON
        return jsonify({'prediction': prediction_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
