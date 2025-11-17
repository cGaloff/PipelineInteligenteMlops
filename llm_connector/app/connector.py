from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
from fastapi import FastAPI
from pydantic import BaseModel

class Message(BaseModel):
    role : str
    content : str

app = FastAPI()

load_dotenv()
genai_api_key = os.getenv("Gemini_API_KEY")

if not genai_api_key:
    raise ValueError("Error al cargar la API KEY")

client = genai.Client(api_key=genai_api_key)

@app.post("/chat")
def generateResponse(messages: list[Message]):

    gemini_messages = []

    for m in messages:
        role_map = 'model' if m.role.lower() == 'assistant' else 'user'
        
        content_object = types.Content(
            role=role_map,
            parts=[types.Part(text=m.content)]
        )
        gemini_messages.append(content_object)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_messages
        )
        answer = response.text
        return {"response": answer.strip()}

    except Exception as e:
        return {"response" : f"error del MLL {e}"}
    
#Run using uvicorn connector:app --reload --port 8000 at llm_connector/app