# client.py
import requests
import sys

def get_compare(text1, file2):
    response = requests.post('http://localhost:5001/compare', 
                             json={'text_list': text1, 'text_file': file2})
    return response.json()['compare']

if __name__ == '__main__':
    if len(sys.argv) > 2:
        sim = get_compare(sys.argv[1], sys.argv[2])
        print(sim)
    else:
        print("Usage: python client.py 'text_list' 'text_file'")
