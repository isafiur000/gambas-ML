# client.py
import requests
import sys

def get_extraction(file1, file2):
    response = requests.post('http://localhost:5001/extraction', 
                             json={'complaint_path': file1, 'symptom_dict': file2})
    return response.json()['extraction']

if __name__ == '__main__':
    if len(sys.argv) > 2:
        sim = get_extraction(sys.argv[1], sys.argv[2])
        print(sim)
    else:
        print("Usage: python client.py 'complaint_path' 'symptom_dict'")
