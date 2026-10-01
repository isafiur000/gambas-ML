import pickle
import pandas as pd
import sys

#Load model
model_path = "/home/safiur/Project/models/ermortality/rf_mortality_model.pickle"
def load_model():
    with open(model_path, "rb") as f:
        return pickle.load(f)
            
# Loading the model
package = load_model()
    
#Get Arguments
resus = sys.argv[1].replace("\n", ", ")
lactate = float(sys.argv[2])
urea = float(sys.argv[3])
creatinine = float(sys.argv[4])
platelets = float(sys.argv[5])


if resus:
    resus_value = resus
else:
    resus = "None"  
    
input_df = pd.DataFrame({
    "Lactate (in ABG)": [lactate],
    "Urea (mg/dl)": [urea],
    "Creatinine (mg/dl)": [creatinine],
    "Platelets (10 ^ 6)": [platelets],
    "Resuscitation Received": [resus_value]
    })  

model = package["model"]
threshold = package["threshold"]

# Getting prediction
proba = model.predict_proba(input_df)[:, 1][0]
prediction = int(proba >= threshold)

print(prediction)

#risk_class = "HIGH RISK" if prediction else "LOW RISK"
#risk_color = "risk-high" if prediction else "risk-low"
