from flask import Flask, request, jsonify
import os
BASE_DIR = os.environ.get('BASE_DIR', '.')
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_VERSION = "0.2" # updated model version
MODEL_PATH = os.path.join(BASE_DIR, 'models/model_v02.joblib')

print(f"Try loading model from: {MODEL_PATH}") # added for debugging

# check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file {MODEL_PATH} not found. Please run train.py first!")
    model = None
else:
    # load the model
    print(f"Try loading model v{MODEL_VERSION}...")
    model = joblib.load(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health():
    # Health check endpoint
    return jsonify({"status": "ok", "model_version": MODEL_VERSION})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded, service unavailable"}), 500

    # 1. Get JSON data
    data = request.get_json()
    
    try:
        # 2. Convert JSON to DataFrame
        features_df = pd.DataFrame([data])
        
        # 3. use the model to make a prediction
        prediction = model.predict(features_df)
        
        # 4. return the prediction as JSON
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        # error handling
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print(f"Starting service at http://localhost:9696")
    app.run(debug=False, host='0.0.0.0', port=9696)