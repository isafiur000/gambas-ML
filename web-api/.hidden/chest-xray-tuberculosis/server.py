# server.py
from flask import Flask, request, jsonify
from pathlib import Path
import torch
import cv2
import sys

app = Flask(__name__)
base_dir = "/home/safiur/Project/models/TBModel"
sys.path.append(base_dir)

from handler import TBClassifier
config_path = Path(base_dir) / "config.json"
model_path = Path(base_dir) / "model.pt"

# Load model immediately when the script starts
print("Loading model...")
classifier = TBClassifier(model_path = model_path, config_path = config_path)
print("Model loaded!")

@app.route('/prediction', methods=['POST'])
def get_prediction():
    data = request.json
    
    image_path = data.get('image_path', '')
    
    image = cv2.imread(image_path)
    result = classifier.predict(image)

    return jsonify({'prediction': result['prediction']})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=False)
