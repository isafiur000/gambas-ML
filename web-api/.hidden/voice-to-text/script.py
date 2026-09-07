from transformers import AutoModelForTDT, AutoProcessor
import torch
import librosa
import sys
from transformers.utils import logging

# Suppress the "Loading weights" and download progress bars
logging.disable_progress_bar()

# (Optional) Set logging to ERROR to silence warning texts
logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/home/safiur/Project/models/audio" # update this model
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForTDT.from_pretrained(model_path, dtype = "auto", device_map = device)

# Load your external audio file
audio_file_path = sys.argv[1]

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
print(text[0])

# https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
