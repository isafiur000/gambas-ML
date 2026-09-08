from sentence_transformers import SentenceTransformer
import sys
from transformers.utils import logging

# Suppress the "Loading weights" and download progress bars
logging.disable_progress_bar()

# (Optional) Set logging to ERROR to silence warning texts
logging.set_verbosity_error()

#Model Directory
model_path = "/home/safiur/Project/models/modern" # Update this

# Load model and processor
nlp_model = SentenceTransformer(model_path)

# Define file path
text1 = sys.argv[1]
text2 = sys.argv[2]

# Create sentence embeddings
embeddings1 = nlp_model.encode(text1)
embeddings2 = nlp_model.encode(text2)

# Calculate cosine similarities
similarities = nlp_model.similarity(embeddings1, embeddings2)

print(similarities.cpu().numpy())

# https://huggingface.co/FremyCompany/BioLORD-2023-C
