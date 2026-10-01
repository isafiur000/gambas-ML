from sentence_transformers import SentenceTransformer, util
import json

# 1. Load the BioLORD model
model_path = '/home/safiur/Project/models/BioLord'
model = SentenceTransformer(model_path)

# 2. Load the symptom dictionary (list of dicts)
with open('/home/safiur/Project/tblsymptoms.json', 'r', encoding='utf-8') as file:
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
patient_complaint = "I have a really bad pounding in my head and I feel like I'm going to throw up."
query_embedding = model.encode(patient_complaint, convert_to_tensor=True)

# 5. Compare patient complaint against symptom definitions
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]

print(f"Patient complaint: '{patient_complaint}'\n")

# 6. Return only the symptom_name for each match
extracted_symptoms = []
for hit in hits:
    symptom = symptom_dictionary[hit['corpus_id']]
    extracted_symptoms.append(symptom['symptom_name'])
    print(f"Extracted Symptom: {symptom['symptom_name']} (Score: {hit['score']:.4f})")

print(f"\nReturned symptom names: {extracted_symptoms}")
