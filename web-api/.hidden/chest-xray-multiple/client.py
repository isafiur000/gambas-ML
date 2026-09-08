# client.py
import requests
import sys

def get_prediction(image_path):
    response = requests.post('http://localhost:5100/prediction', 
                             json={'image_path': image_path})
    return response.json()['prediction']

if __name__ == '__main__':
    if len(sys.argv) > 1:
        predict = get_prediction(sys.argv[1])
        print(predict)
    else:
        print("Usage: python client.py 'image'")
