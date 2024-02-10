
from flask import Flask, request, jsonify
import joblib

xgb = joblib.load('xgboost_model.pkl')

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        # Process the data (perform prediction, etc.)
        prediction = xgb.predict(data)  # Replace this with your actual prediction logic
        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
