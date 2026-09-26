from transformers import AutoModelForCTC, AutoProcessor
import librosa
import torch
import numpy as np

model_id = "/home/safiur/Project/models/audio"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id).to(device)

audio = "/home/safiur/Project/recording_1_final.wav"
speech, sample_rate = librosa.load(audio, sr=16000)
inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True)
inputs = inputs.to(device)

# 1. Get raw logits (NOT generate)
with torch.no_grad():
    logits = model(**inputs).logits

# 2. Argmax to get frame-wise token IDs
frame_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()

# 3. CTC Collapse
collapsed_ids = []
prev_id = None
# The model's special tokens to ignore (blank, bos, eos, unk)
special_ids = {0, 1, 2, 3} 

for token_id in frame_ids:
    token_id = int(token_id)
    # Collapse duplicates AND ignore special tokens
    if token_id != prev_id and token_id not in special_ids:
        collapsed_ids.append(token_id)
    prev_id = token_id

# 4. Decode the cleaned sequence
decoded_text = processor.batch_decode([collapsed_ids], skip_special_tokens=True)[0]
print(decoded_text)

#https://huggingface.co/google/medasr
