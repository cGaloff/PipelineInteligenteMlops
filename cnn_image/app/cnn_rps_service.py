import io
import os
from pathlib import Path
from google import genai
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from dotenv import load_dotenv

# =========================
# Configuración general
# =========================

IMG_SIZE = (160, 160)

# Rutas base
BASE_DIR = Path(__file__).resolve().parents[1]       # cnn_image/
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "cnn_rps.keras"

# Clases tal como las ve el modelo (orden de las carpetas al entrenar)
BACKEND_CLASSES_EN = ["paper", "rock", "scissors"]

# Clases que verá el usuario (en español)
IMAGE_CLASSES = ["piedra", "papel", "tijera"]
EN_TO_ES = {
    "paper": "papel",
    "rock": "piedra",
    "scissors": "tijera",
}
load_dotenv()
gemini_api_key = os.getenv("gemini_api_key")

client = genai.Client(api_key=gemini_api_key)

LIMITATION_MESSAGE = (
    "El sistema solo está entrenado para reconocer imágenes de gestos "
    "de 'piedra', 'papel' y 'tijera' en condiciones similares al dataset "
    "de entrenamiento. Puede fallar con gestos distintos, imágenes muy "
    "oscuras/borrosas o ángulos poco comunes."
)

# =========================
# Cargar modelo al iniciar
# =========================

if not MODEL_PATH.exists():
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}. "
                       "Asegúrate de haber entrenado y guardado cnn_rps.keras")

print(f"Cargando modelo desde: {MODEL_PATH}")
MODEL = keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente.")


# =========================
# Inicializar FastAPI
# =========================

app = FastAPI(
    title="CNN Piedra-Papel-Tijera",
    description="Servicio de clasificación de imágenes usando una CNN preentrenada.",
    version="1.0.0",
)

# (Opcional) habilitar CORS si vas a consumir desde un front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Utilidades
# =========================

def load_image_from_upload(image_bytes: bytes) -> np.ndarray:
    """Carga la imagen subida y la prepara para el modelo (1, 160, 160, 3)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)  # mismo tamaño que en el entrenamiento
        img_np = np.array(img, dtype=np.float32) / 255.0   # normalizar
        img_np = np.expand_dims(img_np, axis=0)            # (1, H, W, 3)
        return img_np
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"No se pudo cargar la imagen: {e}")


def predict_rps(image_array: np.ndarray) -> tuple[str, float]:
    """
    Realiza la predicción con el modelo.
    Devuelve (clase_en_español, confianza).
    """
    preds = MODEL.predict(image_array)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    class_en = BACKEND_CLASSES_EN[pred_idx]
    class_es = EN_TO_ES[class_en]

    return class_es, confidence


# =========================
# Endpoints
# =========================

@app.get("/")
def root():
    return {
        "message": "Servicio CNN Piedra-Papel-Tijera operativo.",
        "classes_supported": IMAGE_CLASSES,
    }


@app.post("/api/cnn/classify")
async def classify_image_endpoint(image: UploadFile = File(...)):
    # Validar tipo de archivo
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=415,
            detail="Tipo de archivo no soportado. Sube una imagen JPEG, PNG o WEBP."
        )

    # Leer bytes
    image_bytes = await image.read()

    try:
        # 1) Cargar y preprocesar imagen
        img_np = load_image_from_upload(image_bytes)

        # 2) Predicción
        prediction, confidence = predict_rps(img_np)

        prompt_explicacion = (
            f"""Eres un analista experto de imagenes.
            interpreta esta predicción {prediction} que tuvo una confianza de {confidence:.2f}. Proporciona una explicación clara, breve y técnica de este resultado.
            No inventes datos. Sé objetivo.
            Retorna la explicación en español.."""
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_explicacion,
        )
        explicacion = response.text

        # 3) Respuesta
        if explicacion:
            return {
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "explanation" : explicacion,
                "classes_supported": IMAGE_CLASSES,
                "limitations": LIMITATION_MESSAGE,
            }

    except HTTPException as e:
        # errores controlados
        raise e
    except Exception as e:
        # errores inesperados
        raise HTTPException(
            status_code=500,
            detail=f"Fallo interno en la clasificación: {e}"
        )
