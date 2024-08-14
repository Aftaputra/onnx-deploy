from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the ONNX model
session = ort.InferenceSession('model.onnx')

# In-memory storage for the data
stored_data = {}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assume input is in JSON format
    input_data = np.array(data['input'], dtype=np.float32)  # Convert input data to numpy array

    # Perform inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    result = float(outputs[0][0][0])  # Convert output to a single float value

    return jsonify({'result': result})

@app.route('/data', methods=['POST'])
def save_data():
    global stored_data
    data = request.json  # Save received JSON data
    timestamp = datetime.now().isoformat()  # Get the current timestamp
    stored_data[timestamp] = data
    return jsonify({'message': 'Data received successfully!', 'timestamp': timestamp})

@app.route('/ambil', methods=['GET'])
def get_data():
    return jsonify(stored_data)  # Send stored data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
