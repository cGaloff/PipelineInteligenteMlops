FROM python:3.10-slim

# No generar .pyc y logs sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar el código del proyecto y el modelo
COPY app ./app
COPY models ./models

# 3. Exponer el puerto del servicio
EXPOSE 8002

# 4. Comando para levantar FastAPI
#    OJO: usamos app.cnn_rps_service:app (archivo cnn_rps_service.py dentro de carpeta app)
CMD ["uvicorn", "app.cnn_rps_service:app", "--host", "0.0.0.0", "--port", "8002"]
