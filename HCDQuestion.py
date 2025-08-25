import requests

response = requests.post("http://localhost:8000/chat", 
                        json={"message": "WHow do you build trust with communities or school actors during a design process?"})
print(response.json()["response"])