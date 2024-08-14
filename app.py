from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from datetime import datetime
import pytz
import requests

app = Flask(__name__)

# Load the ONNX model
session = ort.InferenceSession('model.onnx')

# In-memory storage for the data
stored_data = {}
predicted_data = {}

def format_timestamp():
    # Get current time in GMT+7
    tz = pytz.timezone('Asia/Jakarta')
    now = datetime.now(tz)
    
    # Format timestamp
    formatted_time = now.strftime('%A, %d %B, %H.%M (GMT+7)')
    day_map = {
        'Monday': 'Senin',
        'Tuesday': 'Selasa',
        'Wednesday': 'Rabu',
        'Thursday': 'Kamis',
        'Friday': 'Jumat',
        'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }
    formatted_time = formatted_time.replace(now.strftime('%A'), day_map[now.strftime('%A')])
    
    return formatted_time

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assume input is in JSON format
    input_data = np.array(data['input'], dtype=np.float32)  # Convert input data to numpy array

    # Perform inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    result = float(outputs[0][0][0])  # Convert output to a single float value

    # Get current timestamp
    timestamp = format_timestamp()
    predicted_data[timestamp] = {'input': data['input'], 'result': result}
    
    return jsonify({'result': result, 'timestamp': timestamp})

@app.route('/data', methods=['POST'])
def save_data():
    global stored_data
    data = request.json  # Save received JSON data
    
    # Get the current timestamp
    timestamp = format_timestamp()
    stored_data[timestamp] = data
    
    # Pass data to /predict
    response = request.post('http://onnx-deploy.onrender.com/predict', json={'input': data['input']})
    result = response.json()
    
    # Save the result with timestamp
    predicted_data[timestamp] = {'input': data['input'], 'result': result['result']}
    
    return jsonify({'message': 'Data received and processed successfully!', 'timestamp': timestamp, 'result': result})

@app.route('/ambil', methods=['GET'])
def get_data():
    return jsonify(predicted_data)  # Send predicted data with timestamps

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
