import sys
import json
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging

#Model Directory
model_path = "/home/safiur/Project/models/modern" # Update this

# Suppress the "Loading weights" and download progress bars
logging.disable_progress_bar()

# (Optional) Set logging to ERROR to silence warning texts
logging.set_verbosity_error()

# Load model and processor
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define file path
file_path = sys.argv[1]

# Open the file and read content
with open(file_path, "r") as file:
    paragraph = file.read()

# Tokenize the sentence and add special tokens ([CLS] and [SEP])
inputs = tokenizer(paragraph, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Get the model's hidden states
with torch.no_grad():
    outputs = model(**inputs)

# Extract the [CLS] token embedding (index 0)
# This is a 768-dimensional vector
cls_embedding = outputs.last_hidden_state[:, 0, :]

#print(cls_embedding.shape)  # Output: torch.Size([1, 768])

# Convert to JSON
embedding_list = cls_embedding[0].tolist()  # Convert to list
json_data = json.dumps(embedding_list)

print(json_data)

"""
TO DO IN SERVER SIDE
$ python3 -m venv modern
$ source modern/bin/activate
Then 
$$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
$$ pip install transformers

download trained model files into a folder and update the directory
https://huggingface.co/Simonlee711/Clinical_ModernBERT/tree/main
"""

