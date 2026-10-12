# client.py
import requests
import sys

def get_search(text1):
    response = requests.post('http://localhost:5001/search', 
                             json={'diagnosis_name': text1})
    return response.json()['search']

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sim = get_search(sys.argv[1])
        print(sim)
    else:
        print("Usage: python client.py 'diagnosis_name'")
