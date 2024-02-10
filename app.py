
from flask import Flask, request, jsonify
import joblib
import numpy as np

xgb = joblib.load('xgboost_model.pkl')

app = Flask(_name_)  # Use __name_ instead of name

@app.route('/', methods=['POST'])
def predict():
    try:
        # Convert incoming JSON data to NumPy array
        data = request.json['data']
        if isinstance(data, list):  # Ensure data is in list format
            data = np.array(data)  # Convert list to NumPy array for prediction
        else:
            raise ValueError("Input data is not in list format")

        # Perform prediction
        prediction = xgb.predict(data)

        # Convert NumPy array to list for JSON serialization
        if isinstance(prediction, np.ndarray):
            prediction_list = prediction.tolist()
        else:
            prediction_list = [prediction]

        # Return the predictions as JSON
        return jsonify({'prediction': prediction_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if _name_ == '_main':  # Use __name_ instead of name
    app.run(debug=True)