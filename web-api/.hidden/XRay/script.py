from PIL import Image
import sys
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.utils import logging

#Model Directory
model_path = "/home/safiur/Project/xraymodel" # Update this

# Suppress the "Loading weights" and download progress bars
logging.disable_progress_bar()

# (Optional) Set logging to ERROR to silence warning texts
logging.set_verbosity_error()

# Load model and processor
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)

# Define label columns (class names)
label_columns = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

# Step 1: Load and preprocess the image
image_path = sys.argv[1]  # Replace with your image path

# Open the image
image = Image.open(image_path)

# Ensure the image is in RGB mode (required by most image classification models)
if image.mode != 'RGB':
    image = image.convert('RGB')

# Step 2: Preprocess the image using the processor
inputs = processor(images=image, return_tensors="pt")

# Step 3: Make a prediction (using the model)
with torch.no_grad():  # Disable gradient computation during inference
    outputs = model(**inputs)

# Step 4: Extract logits and get the predicted class index
logits = outputs.logits  # Raw logits from the model
predicted_class_idx = torch.argmax(logits, dim=-1).item()  # Get the class index

# Step 5: Map the predicted index to a class label
# You can also use `model.config.id2label`, but we'll use `label_columns` for this task
predicted_class_label = label_columns[predicted_class_idx]

# Output the results
#print(f"Predicted Class Index: {predicted_class_idx}")
#print(f"Predicted Class Label: {predicted_class_label}")
print(predicted_class_label)


""" 
TO DO IN SERVER SIDE
$ python3 -m venv xray
$ source xray/bin/activate
Then 
$$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
$$ pip install transformers

download trained model files into a folder and update the directory
"""
