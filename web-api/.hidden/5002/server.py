# server.py
from flask import Flask, request, jsonify
from transformers import AutoModelForTDT, AutoProcessor
import torch
import librosa
import sys

app = Flask(__name__)
model_path = "/home/safiur/Project/models/audio" # update this model

# Load model immediately when the script starts
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading model...")
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForTDT.from_pretrained(model_path, dtype = "auto", device_map = device)
print("Model loaded!")

# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.route('/transcribe', methods=['POST'])
def get_transcribe():
    data = request.json
    audio_file_path = data.get('audio_path', '')
    
    # Method 1: Using librosa (recommended)
    audio_array, sampling_rate = librosa.load(audio_file_path, sr = processor.feature_extractor.sampling_rate)
    speech_samples = [audio_array]

    inputs = processor(speech_samples, sampling_rate = processor.feature_extractor.sampling_rate)
    inputs.to(model.device, dtype = model.dtype)

    output = model.generate(
        **inputs, 
        return_dict_in_generate=True,
        max_new_tokens=100  # Adjust this number based on your needs
    )

    text = processor.decode(output.sequences, skip_special_tokens = True)
    
    return jsonify({'transcribe': text[0]})

# ------------------------------------------------------------------
# CLIENT CALL
# ------------------------------------------------------------------       
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
    

