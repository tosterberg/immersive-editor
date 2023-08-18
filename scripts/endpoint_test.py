import requests

URL = "http://localhost:5000"
PREDICT_URL = f"{URL}/predict"
REQUEST = {
    "prompt": "King Edward, be it remembered, was a man of many and varied interests"
}

response = requests.post(PREDICT_URL, data=REQUEST)
print(response)
