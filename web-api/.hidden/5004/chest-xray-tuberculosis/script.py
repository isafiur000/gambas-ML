from pathlib import Path
import torch
import cv2
import sys
from transformers.utils import logging

#Model Directory
base_dir = "/home/safiur/Project/models/TBModel"
sys.path.append(base_dir)

from handler import TBClassifier
config_path = Path(base_dir) / "config.json"
model_path = Path(base_dir) / "model.pt"

image_path = sys.argv[1]  # Replace with your image path

classifier = TBClassifier(model_path = model_path, config_path = config_path)
image = cv2.imread(image_path)
result = classifier.predict(image)
print(result['prediction'])

# https://huggingface.co/sukhmani1303/tuberculosis-vit-model
