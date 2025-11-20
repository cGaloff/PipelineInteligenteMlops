# Proyecto Final MLOps: “Pipeline Inteligente: LLM + ML Clásico + CNN + Despliegue Profesional”

Este proyecto implementa una arquitectura de microservicios orquestada con Docker Compose. Consta de una interfaz de usuario central (Frontend) construida con Gradio que se comunica con tres APIs backend independientes (FastAPI) para realizar tareas de Inteligencia Artificial.

# Arquitectura del Sistema

El sistema se compone de 4 contenedores Docker interconectados:

1. Gradio Frontend (GUI): Interfaz de chat y subida de archivos.

2. Chat LLM Service: API para procesamiento de lenguaje natural.

3. ML Prediction Service: Modelo de random forest (Scikit-Learn) para predicciones de datos de ventas de videojuegos.

4. CNN Image Service: Red Neuronal Convolucional para clasificación de imágenes.

# Ejecución Rápida con Docker

Para levantar todo el sistema automáticamente:

Es necesario tener Docker.

Clona el repositorio.

Ejecuta en la raíz del proyecto:

``` docker-compose up --build ```

Accede a la interfaz gráfica en: http://localhost:7860

# Endpoints
Gradio Frontend: http://localhost:7860

LLM Connector API: http://localhost:8000/chat

ML scikit-learn: http://localhost:8003/model/linear/prediction

Clasificador Imagen: http://localhost:8002/api/cnn/classify

MLflow: http://localhost:5000