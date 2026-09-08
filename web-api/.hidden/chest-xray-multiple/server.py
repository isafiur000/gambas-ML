from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import sys

app = Flask(__name__)
model_path = "/home/safiur/Project/models/xraymodel" # Update this

# Load model immediately when the script starts
print("Loading model...")
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
print("Model loaded!")

# Define label columns (class names)
label_columns = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

@app.route('/prediction', methods=['POST'])
def get_prediction():
    data = request.json
    
    # Step 1: Load and preprocess the image
    image_path = data.get('image_path', '')
    
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
    
    #Step 5: Map the predicted index to a class label
    predicted_class_label = label_columns[predicted_class_idx]
    
    return jsonify({'prediction': predicted_class_label})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=False)
