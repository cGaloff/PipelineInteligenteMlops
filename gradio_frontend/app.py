import gradio as gr
import requests
from dotenv import load_dotenv
import os


load_dotenv()
url = os.getenv("url")
cnn_url = os.getenv("CNN_URL")

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
    

def cnn_image(image_filepath):
    if image_filepath is None:
        return "Suba una imagen"
    try:
        with open(image_filepath, "rb") as f:
            files = {'image' : (image_filepath, f, "image/jpeg")}

            response = requests.post(cnn_url, files=files)
            response.raise_for_status()

            data = response.json()

            if data.get("result") == "Error":
                return f"Error: {data.get('message')}"
            
            prediction = data.get("prediction")
            confidence = data.get("confidence")
            limitations = data.get("limitations")

            output_text = (
                f"**Resultado de la Clasificación:** {prediction.upper()}\n"
                f"**Confianza:** {confidence}\n\n"
                f"**Limitaciones del Sistema:**\n{limitations}"
            )
            return output_text
        
    except requests.exceptions.RequestException as e:
        return f"Error de conexión con el servicio CNN: {e}"


with gr.Blocks() as interface:
    gr.Markdown('# Chat LLM, análisis con ML y clasificación de imágenes')

    gr.ChatInterface(chat, title="Chat LLM")

    gr.Markdown('# Análisis con ML')
    inputs=gr.File(label="Sube el archivo con datos", type="filepath", file_count='single'),
    sendFile = gr.Button("Enviar")
    
    gr.Markdown('# Clasificación de imagenes')
    image = gr.File(label="Sube tu imagen", type="filepath", file_count='single', file_types=['image'])
    image_output = gr.TextArea(label="Resultados", interactive=False)
    sendImage = gr.Button("Enviar")

    sendImage.click(
        fn=cnn_image,
        inputs=[image],
        outputs=[image_output]
    )

interface.launch()