# server.py
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
from tensorflow.image import resize as tf_resize
import torch
import sys

app = Flask(__name__)
model_path = "/home/safiur/Project/models/imagematch" # update this model

# Load model immediately when the script starts
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading model...")
model = AutoModel.from_pretrained(model_path).to(device)
processor = AutoProcessor.from_pretrained(model_path)
print("Model loaded!")

#resizing operation with `tf.image.resize` to match the implementation
def resize(image):
    return Image.fromarray(
        tf_resize(
            images=image, size=[448, 448], method='bilinear', antialias=False
        ).numpy().astype(np.uint8)
    )


@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.json
    img_path = data.get('img_path', '')
    text_list = data.get('text_list', '')
    
    imgs = [Image.open(img_path).convert("RGB")]
    texts = text_list.strip('"').split("\\n")
    
    resized_imgs = [resize(img) for img in imgs]
    inputs = processor(text=texts, images=resized_imgs, padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits_per_image = outputs.logits_per_image
    probs = torch.softmax(logits_per_image, dim=1)
    
    lists = []
    for n_img, img in enumerate(imgs):
        for i, label in enumerate(texts):
            lists.append(f"{label} : Probablity = {probs[n_img][i]:.2%}")
            #print(f"'{label}' Probablity = {probs[n_img][i]:.2%}")


    return jsonify({'similarity': "\n".join(lists)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=False)
            

    

