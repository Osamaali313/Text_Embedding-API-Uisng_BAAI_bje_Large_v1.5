import requests

response = requests.post("http://127.0.0.1:8000/predict", json={"input": "What is the capital of France?"})
print(response.json())
