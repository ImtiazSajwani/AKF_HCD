# AKF_HCD
AKF Human Center Design Teacher model 

Minimum of 16GB of Nvidia GPU memory is required, it will run on CPU but will be extremely slow!!!

This simple API server lets any chat application talk to your trained HCD expert model. Just send a question, get expert HCD advice back!
1. Clone the github.
1. Install all the required modules using pip: pip -r install requirements.txt
2. Make sure the model is located at AKF_HCD_Teacher_Model
3. Run the server using uvicorn server 
4. Chat API (Post the requests and get the response back)
response = requests.post("http://localhost:8000/chat", 
                        json={"message": "WHow do you build trust with communities or school actors during a design process?"})
print(response.json()["response"])
5. This model only serves the HCD related questions others are ignored.
see example for non-HCD
response = requests.post("http://localhost:8000/chat", 
                        json={"message": "What's the weather like today?"})
print(response.json()["response"])

API
/chat POST
/ask-expert POST
/model-info GET
/health GET

Python Example
import requests

def ask_hcd_expert(question):
    response = requests.post("http://localhost:8000/ask-expert", 
                           json={"message": question})
    return response.json()["response"]

# Use it
answer = ask_hcd_expert("Question?")
print(answer)

Javascript:
