# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # Load your machine learning model
# # Replace this with your actual model loading code
# def predict(data):
#     # Your machine learning model prediction code goes here
#     # This is just a placeholder
#     return {"prediction": "your_prediction"}

# @app.route('/', methods=['POST'])



# def handle_prediction():
#     data = request.json['data']
#     prediction = predict(data)
#     return jsonify(prediction)

# if __name__ == '__main__':
#     app.run(debug=True)


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
