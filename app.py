# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# xgb = joblib.load('xgboost_model.pkl')

# app = Flask(__name__)  # Use _name instead of name

# @app.route('/', methods=['POST'])
# def predict():
#     try:
#         # Convert incoming JSON data to NumPy array
#         data = request.json['data']
#         if isinstance(data, list):  # Ensure data is in list format
#             data = np.array(data)  # Convert list to NumPy array for prediction
#         else:
#             raise ValueError("Input data is not in list format")

#         # Perform prediction
#         prediction = xgb.predict(data)

#         # Convert NumPy array to list for JSON serialization
#         if isinstance(prediction, np.ndarray):
#             prediction_list = prediction.tolist()
#         else:
#             prediction_list = [prediction]

#         # Return the predictions as JSON
#         return jsonify({'prediction': prediction_list}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == 'main':  # Use __name instead of name
#     app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import joblib
# from tensorflow.keras.models import load_model
# from pydub import AudioSegment

# Load the trained model
xgb = joblib.load('xgboost_model.pkl')

# Function to preprocess the audio file
# def preprocess_audio(audio_file):
#     # Convert audio to wav format (if not already in wav)
#     sound = AudioSegment.from_file(audio_file)
#     if sound.channels != 1:
#         sound = sound.set_channels(1)
#     if sound.frame_rate != 16000:
#         sound = sound.set_frame_rate(16000)
#     if sound.sample_width != 2:
#         sound = sound.set_sample_width(2)

#     # Ensure uniform length of audio (if required)
#     target_length = 16000  # Example target length
#     sound = sound.set_frame_rate(16000)
#     sound = sound.set_sample_width(2)
#     sound = sound.set_channels(1)
#     sound = sound.set_frame_rate(target_length)
#     sound = sound.set_channels(1)
#     sound = sound.set_sample_width(2)
#     sound = sound.set_frame_rate(target_length)

#     # Export the processed audio to a temporary file
#     temp_audio_file = 'processed_audio.wav'
#     sound.export(temp_audio_file, format='wav')
    
#     return temp_audio_file

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
        #processed_audio_file = preprocess_audio(audio_file)
        
        # Perform prediction using the loaded model
        #prediction = model.predict(processed_audio_file)
        # Note: You need to adapt this part to fit your model's input requirements
        
        # Dummy prediction for testing
        prediction = 1
        
        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
