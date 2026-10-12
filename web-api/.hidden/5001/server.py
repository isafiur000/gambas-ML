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
    if isinstance(s, dict) and s.get("code") and s.get("definition")
]
print(f"Loaded {len(symptom_dictionary)} symptoms.")

print("Precomputing corpus embeddings from symptom definitions...")
symptom_texts = [s['definition'] for s in symptom_dictionary]
with torch.no_grad():
    symptom_embeddings = model.encode(
        symptom_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
print(f"Symptoms embeddings ready: shape = {tuple(symptom_embeddings.shape)}")

# ------------------------------------------------------------------
# LOAD DIAGNOSIS DICTIONARY EMBEDDINGS
# -----------------------------------------------------------------
DIAGNOSIS_DICT_PATH = "/etc/web-api/5001/search_diagnosis/tblicdcomplete.json"

print("Loading diagnosis definitions...")
with open(DIAGNOSIS_DICT_PATH, 'r', encoding='utf-8') as file:
    diagnosis_dictionary = json.load(file)

# Filter out incomplete entries
diagnosis_dictionary = [
    s for s in diagnosis_dictionary
    if isinstance(s, dict) and s.get("code") and s.get("definition")
]
print(f"Loaded {len(diagnosis_dictionary)} diagnosis.")

print("Precomputing corpus embeddings from diagnosis definitions...")
diagnosis_texts = [s['definition'] for s in diagnosis_dictionary]
with torch.no_grad():
    diagnosis_embeddings = model.encode(
        diagnosis_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
print(f"Diagnosis embeddings ready: shape = {tuple(diagnosis_embeddings.shape)}")

# ------------------------------------------------------------------
# Warm-up so the first real request isn't slow
# ------------------------------------------------------------------
with torch.no_grad():
    _ = model.encode("warmup", convert_to_tensor=True, normalize_embeddings=True)
print("Warm-up complete. Server is ready.")


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
# 1. Text Similarity
@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    sim = model.similarity(emb1, emb2)
    
    return jsonify({'similarity': float(sim.cpu().numpy()[0][0])})

# 2. Compare paragraph with a list of differential diagnosis
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

# 3. Extract symptoms from paragraph
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
        symptom_embeddings,
        top_k=len(symptom_dictionary),
    )[0]

    extracted_symptoms = []
    for hit in hits:
        if hit['score'] > SCORE_THRESHOLD:
            symptom = symptom_dictionary[hit['corpus_id']]
            extracted_symptoms.append(
                f"{symptom['code']} Score: {hit['score']:.4f}"
            )

    return jsonify({'extraction': "\n".join(extracted_symptoms)})

# 4. Search appropriate ICD diagnosis from given free text
@app.route('/search', methods=['POST'])
def get_search():
    data = request.json or {}
    patient_diagnosis = data.get('diagnosis_name', '')

    # Encode ONLY the query
    with torch.no_grad():
        diagno_embedding = model.encode(
            patient_diagnosis,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    # Compare against ALL symptom definitions so the threshold sees everything
    matches = util.semantic_search(
        diagno_embedding,
        diagnosis_embeddings,
        top_k=len(diagnosis_dictionary),
    )[0]

    matching_diagnosis = []
    for match in matches:
        if match['score'] > SCORE_THRESHOLD:
            diagnosis = diagnosis_dictionary[match['corpus_id']]
            matching_diagnosis.append(
                f"{diagnosis['definition']} : {diagnosis['code']} Score: {match['score']:.4f}"
            )

    return jsonify({'search': "\n".join(matching_diagnosis)})

# ------------------------------------------------------------------
# CLIENT CALL
# ------------------------------------------------------------------   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

