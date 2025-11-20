# Detalle de Servicios
1. Gradio Frontend (GUI)

Puerto: ```7860```

# Propósito

Provee la interfaz visual para que el usuario interactúe con los modelos. No procesa datos pesados, solo orquesta las peticiones a los otros servicios.

# Variables de Entorno

Estas variables son críticas para que Gradio sepa dónde encontrar a los otros contenedores.

url = http://servicio-chat:8000/chat

model_urlhttp://servicio-linear:8003/model/linear/prediction

CNN_URLhttp://servicio-cnn:8002/api/cnn/classify

# Ejecución Local (Sin Docker)
``` 1. Instalar dependencias
pip install -r requirements.txt

export url="http://localhost:8000/chat"
export model_url="http://localhost:8003/model/linear/prediction"
export CNN_URL="http://localhost:8002/api/cnn/classify"

python app.py