# client.py
import requests
import sys

def get_extraction(file1):
    response = requests.post('http://localhost:5001/extraction', 
                             json={'complaint_path': file1})
    return response.json()['extraction']

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sim = get_extraction(sys.argv[1])
        print(sim)
    else:
        print("Usage: python client.py 'complaint_path'")
