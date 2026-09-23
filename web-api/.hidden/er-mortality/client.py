# client.py
import requests
import sys

def get_prediction(resus, lactate, urea, creatinine, platelets):
    response = requests.post('http://localhost:5006/prediction', json={'resus': resus, 'lactate': lactate, 'urea': urea, 'creatinine': creatinine, 'platelets': platelets})
    return response.json()['prediction']

if __name__ == '__main__':
    if len(sys.argv) > 5:
        pred = get_prediction(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        print(pred)
    else:
        print("Usage: python client.py 'resus' 'lactate' 'urea' 'creatinine platelets'")
