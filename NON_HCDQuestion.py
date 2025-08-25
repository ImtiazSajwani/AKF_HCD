import requests

response = requests.post("http://localhost:8000/chat", 
                        json={"message": "What's the weather like today?"})
print(response.json()["response"])