# server.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
model_path = "/home/safiur/Project/models/BioLord"

# Load model immediately when the script starts
print("Loading model...")
model = SentenceTransformer(model_path)
print("Model loaded!")

@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    sim = model.similarity(emb1, emb2)
    
    return jsonify({'similarity': float(sim.cpu().numpy()[0][0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

