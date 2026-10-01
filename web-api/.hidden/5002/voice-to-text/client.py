# client.py
import requests
import sys

def get_transcribe(audio_path):
    response = requests.post('http://localhost:5002/transcribe', 
                             json={'audio_path': audio_path})
    return response.json()['transcribe']

if __name__ == '__main__':
    if len(sys.argv) > 1:
        predict = get_transcribe(sys.argv[1])
        print(predict)
    else:
        print("Usage: python client.py 'audio'")
