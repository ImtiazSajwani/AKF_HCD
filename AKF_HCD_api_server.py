#!/usr/bin/env python3
"""
HCD Expert Teacher Model API
Specialized model trained specifically for Human-Centered Design guidance
!!Make sure to pip install -r requirements.txt!!

"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
import os
import json

# Request/response models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Global variables for the HCD expert model
hcd_expert_model = None
hcd_tokenizer = None

# Create FastAPI app
app = FastAPI(title="HCD Expert Teacher Model API")

def check_model_path(model_path):
    """Check if model path exists and has required files"""
    if not os.path.exists(model_path):
        print(f" Model path does not exist: {model_path}")
        return False
    
    required_files = ['adapter_config.json', 'adapter_model.safetensors']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            print(f" Missing required file: {file} in {model_path}")
            return False
    
    print(f" Model path and files found: {model_path}")
    return True

def load_hcd_expert_model():
    """Load the specialized HCD expert model with error handling"""
    global hcd_expert_model, hcd_tokenizer
    
    print("Loading HCD Expert Teacher Model...")
    
    # YOUR TRAINED MODEL PATH - Change this to your model location
    HCD_MODEL_PATH = "./AKF_HCD_Teacher_Model"
    
    # Check if model exists
    if not check_model_path(HCD_MODEL_PATH):
        print("Please check your model path and try again.")
        return False
    
    try:
        # Load tokenizer first
        print("Loading tokenizer...")
        hcd_tokenizer = AutoTokenizer.from_pretrained(HCD_MODEL_PATH)
        if hcd_tokenizer.pad_token is None:
            hcd_tokenizer.pad_token = hcd_tokenizer.eos_token
        print(" Tokenizer loaded successfully!")
        
        # Read adapter config to get base model info
        config_path = os.path.join(HCD_MODEL_PATH, 'adapter_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
            print(f"Using base model from config: {base_model_name}")
        else:
            base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            print(f"Using default base model: {base_model_name}")
        
        # Try different loading approaches
        print("Loading base model...")
        
        # Method 1: Load with BitsAndBytesConfig for memory efficiency
        try:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            print(" Base model loaded with quantization")
            
        except Exception as e:
            print(f"Quantized loading failed: {e}")
            print("Trying standard loading...")
            
            # Method 2: Standard loading
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            print(" Base model loaded successfully!")
        
        # Load the adapter
        print("Loading your HCD expertise adapter...")
        hcd_expert_model = PeftModel.from_pretrained(
            base_model, 
            HCD_MODEL_PATH,
            torch_dtype=torch.bfloat16,
        )
        hcd_expert_model.eval()
        
        print(" HCD Expert Teacher Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f" Error loading model!!!!!: {str(e)}")

        return False

def is_hcd_related(message: str) -> bool:
    """Check if the message is related to HCD topics"""
    hcd_keywords = [
        'design', 'hcd', 'human-centered', 'user', 'prototype', 'empathy', 
        'facilitat', 'workshop', 'co-design', 'participat', 'stakeholder',
        'trust', 'community', 'bias', 'power dynamic', 'iteration', 'feedback',
        'test', 'interview', 'observation', 'insight', 'solution', 'problem',
        'inclusive', 'equity', 'collaboration', 'creative', 'innovation',
        'mindset', 'process', 'method', 'tool', 'framework', 'approach',
        'research', 'discover', 'define', 'ideate', 'build', 'learn',
        'education', 'school', 'teacher', 'student', 'learning'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in hcd_keywords)

def generate_hcd_response(user_message: str) -> str:
    """Generate specialized HCD response"""
    
    # Check if question is HCD-related
    if not is_hcd_related(user_message):
        return """I'm a specialized Human-Centered Design expert. I focus specifically on HCD methodology, facilitation, design processes, building trust with communities, managing power dynamics, prototyping, and educational design challenges. 

Could you ask me something related to Human-Centered Design? For example:
- How to build trust in design processes
- Managing power dynamics in co-design sessions  
- Facilitation techniques for inclusive workshops
- Prototyping and testing with users
- Implementing HCD in educational settings"""

    # Create specialized HCD expert persona
    system_prompt = """You are a specialized Human-Centered Design expert and teacher. You have deep experience in:

- Facilitating inclusive design processes in educational and community settings
- Building trust and managing relationships with stakeholders
- Handling power dynamics and bias in co-design sessions
- Developing and testing prototypes with real users
- Training others in HCD methodology and mindsets
- Implementing equity and inclusion in design work
- Supporting teams through the emotional aspects of design challenges

Provide thoughtful, practical, and empathetic guidance. Always ground your advice in real-world experience and encourage reflection. Ask clarifying questions when helpful."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # Generate using your trained HCD expertise
        formatted_text = hcd_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = hcd_tokenizer(formatted_text, return_tensors="pt").to(hcd_expert_model.device)
        
        with torch.no_grad():
            outputs = hcd_expert_model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=hcd_tokenizer.eos_token_id,
                eos_token_id=hcd_tokenizer.eos_token_id,
            )
        
        # Extract the expert response
        response = hcd_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            expert_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            expert_response = response[len(formatted_text):].strip()
        
        return expert_response
        
    except Exception as e:
        return f"I encountered a technical issue while generating a response: {str(e)}. Please try asking your question in a different way."

@app.on_event("startup")
async def startup_event():
    """Load the HCD expert model when server starts"""
    success = load_hcd_expert_model()
    if not success:
        print(" Failed to load model. Server will start but responses will be limited.")

@app.get("/")
async def root():
    """API information"""
    return {
        "model": "HCD Expert Teacher Model",
        "description": "Specialized AI trained exclusively for Human-Centered Design guidance",
        "status": "ready" if hcd_expert_model is not None else "model_loading_failed",
        "expertise": [
            "HCD methodology and processes",
            "Facilitation and workshop design", 
            "Building trust with communities",
            "Managing power dynamics and bias",
            "Prototyping and user testing",
            "Educational design challenges",
            "Equity and inclusion in design"
        ],
        "usage": "POST to /ask-expert with {'message': 'your HCD question'}"
    }

@app.post("/ask-expert", response_model=ChatResponse)
async def ask_hcd_expert(request: ChatMessage):
    """
    Ask the HCD Expert Teacher
    
    This is a specialized model trained specifically on Human-Centered Design
    Send your HCD questions and get expert guidance!
    """
    
    if hcd_expert_model is None:
        return ChatResponse(
            response="I'm sorry, but the HCD Expert Model failed to load properly. Please check the server logs and ensure your model files are accessible. You can still ask HCD questions and I'll try to provide basic guidance, but the specialized responses may not be available."
        )
    
    try:
        # Get expert response from your trained model
        expert_response = generate_hcd_response(request.message)
        return ChatResponse(response=expert_response)
    
    except Exception as e:
        return ChatResponse(response=f"I encountered a technical issue: {str(e)}. Please try rephrasing your question.")

@app.post("/chat", response_model=ChatResponse)  # Keep this for compatibility
async def chat(request: ChatMessage):
    """Legacy endpoint - redirects to ask-expert"""
    return await ask_hcd_expert(request)

@app.get("/model-info")
async def model_info():
    """Information about this specialized model"""
    return {
        "model_name": "HCD Expert Teacher Model",
        "specialization": "Human-Centered Design",
        "training_focus": "Educational and community design processes",
        "capabilities": [
            "Design process facilitation",
            "Stakeholder engagement strategies", 
            "Power dynamics management",
            "Inclusive design practices",
            "Prototyping guidance",
            "Trust building techniques",
            "Workshop design and facilitation"
        ],
        "status": "ready" if hcd_expert_model is not None else "failed_to_load"
    }

@app.get("/health")
async def health():
    """Check if the HCD expert model is ready"""
    return {
        "status": "ready" if hcd_expert_model is not None else "model_unavailable",
        "model": "HCD Expert Teacher Model",
        "specialization": "Human-Centered Design"
    }

if __name__ == "__main__":
    print(" Starting HCD Expert Teacher Model...")
    print(" This is a specialized model trained for Human-Centered Design")
    print(" API will be available at: http://localhost:8000")
    
    uvicorn.run(
        "AKF_HCD_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )