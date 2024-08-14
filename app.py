from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the ONNX model
session = ort.InferenceSession('model.onnx')

# In-memory storage for the data
stored_data = {}
predicted_data = {}

def get_timestamp():
    # Get the current timestamp in ISO format
    return datetime.now().isoformat()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array(data['input'], dtype=np.float32)
        
        # Perform inference
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})
        result = float(outputs[0][0][0])
        
        timestamp = get_timestamp()
        predicted_data[timestamp] = {'input': data['input'], 'result': result}
        
        return jsonify({'result': result, 'timestamp': timestamp})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data', methods=['POST'])
def save_data():
    try:
        data = request.json
        timestamp = get_timestamp()
        stored_data[timestamp] = data
        
        # Pass data to /predict
        response = requests.post('http://onnx-deploy.onrender.com/predict', json={'input': data['input']})
        result = response.json()
        
        # Save the result with timestamp
        predicted_data[timestamp] = {'input': data['input'], 'result': result['result']}
        
        return jsonify({'message': 'Data received and processed successfully!', 'timestamp': timestamp, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ambil', methods=['GET'])
def get_data():
    return jsonify(predicted_data)  # Send predicted data with timestamps

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
