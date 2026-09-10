# server.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

model_path = "/etc/web-api/er-mortality/rf_mortality_model.pickle"
def load_model():
    with open(model_path, "rb") as f:
        return pickle.load(f)
        
# Load model immediately when the script starts
print("Loading model...")
package = load_model()
print("Model loaded!")

@app.route('/prediction', methods=['POST'])
def get_prediction():
    data = request.json
    
    resus = data.get('resus', 'None').replace("\n", ", ")
    lactate = float(data.get('lactate'))
    urea = float(data.get('urea'))
    creatinine = float(data.get('creatinine'))
    platelets = float(data.get('platelets')) 
        
    input_df = pd.DataFrame({
        "Lactate (in ABG)": [lactate],
        "Urea (mg/dl)": [urea],
        "Creatinine (mg/dl)": [creatinine],
        "Platelets (10 ^ 6)": [platelets],
        "Resuscitation Received": [resus]
    })
    
    model = package["model"]
    threshold = package["threshold"]
    
    # Getting prediction
    proba = model.predict_proba(input_df)[:, 1][0]
    prediction = int(proba >= threshold)
    
    return jsonify({'prediction': prediction})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5250, debug=False)
    
    
    
