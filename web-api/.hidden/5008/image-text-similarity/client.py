# client.py
import requests
import sys

def get_similarity(text1, file2):
    response = requests.post('http://localhost:5008/similarity', 
                             json={'text_list': text1, 'img_path': file2})
    return response.json()['similarity']

if __name__ == '__main__':
    if len(sys.argv) > 2:
        sim = get_similarity(sys.argv[1], sys.argv[2])
        print(sim)
    else:
        print("Usage: python client.py 'text_list' 'img_path'")
