import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
from tensorflow.image import resize as tf_resize
import torch
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/home/safiur/Project/models/imagematch"
model = AutoModel.from_pretrained(model_path).to(device)
processor = AutoProcessor.from_pretrained(model_path)

#Get File and semicolon separated list of options
img_path = sys.argv[1]
imgs = [Image.open(img_path).convert("RGB")]

text_list = sys.argv[2]
texts = text_list.split(";")

# If you want to reproduce the results from MedSigLIP evals, we recommend a
# resizing operation with `tf.image.resize` to match the implementation with the
# Big Vision library (https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/pp/ops_image.py#L84).
# Otherwise, you can rely on the Transformers image processor's built-in
# resizing (done automatically by default and uses `PIL.Image.resize`) or use
# another resizing method.
def resize(image):
    return Image.fromarray(
        tf_resize(
            images=image, size=[448, 448], method='bilinear', antialias=False
        ).numpy().astype(np.uint8)
    )


resized_imgs = [resize(img) for img in imgs]

inputs = processor(text=texts, images=resized_imgs, padding="max_length", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.softmax(logits_per_image, dim=1)

for n_img, img in enumerate(imgs):
    for i, label in enumerate(texts):
        print(f"'{label}' Probablity = {probs[n_img][i]:.2%}")

# Get the image and text embeddings
#print(f"image embeddings: {outputs.image_embeds}")
#print(f"text embeddings: {outputs.text_embeds}")

# https://huggingface.co/google/medsiglip-448

