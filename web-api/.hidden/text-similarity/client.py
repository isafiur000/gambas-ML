# client.py
import requests
import sys

def get_similarity(text1, text2):
    response = requests.post('http://localhost:5001/similarity', 
                             json={'text1': text1, 'text2': text2})
    return response.json()['similarity']

if __name__ == '__main__':
    if len(sys.argv) > 2:
        sim = get_similarity(sys.argv[1], sys.argv[2])
        print(sim)
    else:
        print("Usage: python client.py 'text1' 'text2'")
