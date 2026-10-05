# server.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import torch

app = Flask(__name__)
MODEL_PATH = "/home/safiur/Project/models/BioLord"

print("Loading BioLORD model...")
model = SentenceTransformer(MODEL_PATH)
print("Model loaded!")

# ------------------------------------------------------------------
# LOAD SYMPTOMS DICTIONARY EMBEDDINGS
# -----------------------------------------------------------------
SYMPTOM_DICT_PATH = "/etc/web-api/5001/extract_symptoms/tblsymptoms.json"
SCORE_THRESHOLD = 0.5

print("Loading symptom definitions...")
with open(SYMPTOM_DICT_PATH, 'r', encoding='utf-8') as file:
    symptom_dictionary = json.load(file)

# Filter out incomplete entries
symptom_dictionary = [
    s for s in symptom_dictionary
    if isinstance(s, dict) and s.get("symptom_name") and s.get("symptom_definition")
]
print(f"Loaded {len(symptom_dictionary)} symptoms.")

print("Precomputing corpus embeddings from symptom definitions...")
corpus_texts = [s['symptom_definition'] for s in symptom_dictionary]
with torch.no_grad():
    corpus_embeddings = model.encode(
        corpus_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
print(f"Corpus embeddings ready: shape = {tuple(corpus_embeddings.shape)}")

# Warm-up so the first real request isn't slow
with torch.no_grad():
    _ = model.encode("warmup", convert_to_tensor=True, normalize_embeddings=True)
print("Warm-up complete. Server is ready.")


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    sim = model.similarity(emb1, emb2)
    
    return jsonify({'similarity': float(sim.cpu().numpy()[0][0])})
    
@app.route('/compare', methods=['POST'])
def get_compare():
    data = request.json
    text_list = data.get('text_list', '')
    text_file = data.get('text_file', '')
    
    texts = text_list.strip('"').split("\\n")
    
    with open(text_file, 'r', encoding="utf-8") as file:
        text_data = file.read()
    
    
    emb1 = model.encode(text_data)
    lists = []
    for i, label in enumerate(texts):
        emb2 = model.encode(label)
        sim = model.similarity(emb1, emb2)
        proba = float(sim.cpu().numpy()[0][0])
        lists.append(f"{label} : Probablity = {proba:.2%}")
            
    return jsonify({'compare': "\n".join(lists)})

@app.route('/extraction', methods=['POST'])
def get_extraction():
    data = request.json or {}
    complaint_path = data.get('complaint_path', '')

    # Load patient complaint text file
    with open(complaint_path, 'r', encoding="utf-8") as file:
        patient_complaint = file.read()

    # Encode ONLY the query — corpus embeddings are already in memory
    with torch.no_grad():
        query_embedding = model.encode(
            patient_complaint,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    # Compare against ALL symptom definitions so the threshold sees everything
    hits = util.semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k=len(symptom_dictionary),
    )[0]

    extracted_symptoms = []
    for hit in hits:
        if hit['score'] > SCORE_THRESHOLD:
            symptom = symptom_dictionary[hit['corpus_id']]
            extracted_symptoms.append(
                f"{symptom['symptom_name']} Score: {hit['score']:.4f}"
            )

    return jsonify({'extraction': "\n".join(extracted_symptoms)})

# ------------------------------------------------------------------
# CLIENT CALL
# ------------------------------------------------------------------   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

