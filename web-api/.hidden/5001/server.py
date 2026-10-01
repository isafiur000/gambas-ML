# server.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json

app = Flask(__name__)
model_path = "/home/safiur/Project/models/BioLord"

# Load model immediately when the script starts
print("Loading model...")
model = SentenceTransformer(model_path)
print("Model loaded!")

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
    data = request.json
    complaint_path = data.get('complaint_path', '')
    symptom_dict = data.get('symptom_dict', '')
    
    #load patient complaints textfile
    with open(complaint_path, 'r', encoding="utf-8") as file:
        patient_complaint = file.read()
    
    #load symptoms dict json file
    with open(symptom_dict, 'r', encoding='utf-8') as file:
        symptom_dictionary = json.load(file)
        
    # Optional: filter out incomplete entries
    symptom_dictionary = [
        s for s in symptom_dictionary
        if isinstance(s, dict) and s.get("symptom_name") and s.get("symptom_definition")
    ]
    
    # 3. Create corpus embeddings from symptom DEFINITIONS only
    corpus_texts = [s['symptom_definition'] for s in symptom_dictionary]
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

    # 4. Process a patient complaint
    query_embedding = model.encode(patient_complaint, convert_to_tensor=True)
    
    # 5. Compare patient complaint against symptom definitions
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
    
    # 6. Return only the symptom_name for each match
    SCORE_THRESHOLD = 0.5
    extracted_symptoms = []
    for hit in hits:
        if hit['score'] > SCORE_THRESHOLD:
            symptom = symptom_dictionary[hit['corpus_id']]
            #extracted_symptoms.append(symptom['symptom_name'])
            extracted_symptoms.append(f"{symptom['symptom_name']} Score: {hit['score']:.4f}")
        
    return jsonify({'extraction': "\n".join(extracted_symptoms)})
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

