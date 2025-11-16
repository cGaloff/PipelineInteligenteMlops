import gradio as gr
import requests
from dotenv import load_dotenv
import os


load_dotenv()
url = os.getenv("url")

def chat(message, history):
    messages = []

    for msg, ai_msg in history:
        messages.append({"role" : "user", "content" : msg})
        messages.append({"role" : "model", "content" : ai_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = requests.post(url, json=messages)
        response.raise_for_status()

        data = response.json()
        ai_response = data.get("response", "Error al obtener la respuesta")

        return ai_response
    except requests.exceptions.RequestException as e:
        return f"Error {e}"

with gr.Blocks() as interface:
    gr.Markdown('# Chat LLM, análisis con ML y clasificación de imágenes')

    gr.ChatInterface(chat, title="Chat LLM")

    gr.Markdown('# Análisis con ML')
    inputs=gr.File(label="Sube el archivo con datos", type="filepath", file_count='single'),
    sendFile = gr.Button("Enviar")
    
    gr.Markdown('# Clasificación de imagenes')
    image = gr.File(label="Sube tu imagen", type="filepath", file_count='single', file_types=['image'])
    sendImage = gr.Button("Enviar")

interface.launch()