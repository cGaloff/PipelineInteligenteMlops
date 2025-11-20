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

IMG_SIZE = (160, 160)

BASE_DIR = Path(__file__).resolve().parents[1]       # cnn_image/
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "cnn_rps.keras"

BACKEND_CLASSES_EN = ["paper", "rock", "scissors"]


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


if not MODEL_PATH.exists():
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}. "
                       "Asegúrate de haber entrenado y guardado cnn_rps.keras")

print(f"Cargando modelo desde: {MODEL_PATH}")
MODEL = keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente.")



app = FastAPI(
    title="CNN Piedra-Papel-Tijera",
    description="Servicio de clasificación de imágenes usando una CNN preentrenada.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def load_image_from_upload(image_bytes: bytes) -> np.ndarray:
    
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)  
        img_np = np.array(img, dtype=np.float32) / 255.0   
        img_np = np.expand_dims(img_np, axis=0)            
        return img_np
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"No se pudo cargar la imagen: {e}")


def predict_rps(image_array: np.ndarray) -> tuple[str, float]:

    preds = MODEL.predict(image_array)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    class_en = BACKEND_CLASSES_EN[pred_idx]
    class_es = EN_TO_ES[class_en]

    return class_es, confidence


@app.get("/")
def root():
    return {
        "message": "Servicio CNN Piedra-Papel-Tijera operativo.",
        "classes_supported": IMAGE_CLASSES,
    }


@app.post("/api/cnn/classify")
async def classify_image_endpoint(image: UploadFile = File(...)):
    
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=415,
            detail="Tipo de archivo no soportado. Sube una imagen JPEG, PNG o WEBP."
        )

   
    image_bytes = await image.read()

    try:
       
        img_np = load_image_from_upload(image_bytes)

        
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

        
        if explicacion:
            return {
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "explanation" : explicacion,
                "classes_supported": IMAGE_CLASSES,
                "limitations": LIMITATION_MESSAGE,
            }

    except HTTPException as e:
        raise e
    except Exception as e:
        
        raise HTTPException(
            status_code=500,
            detail=f"Fallo interno en la clasificación: {e}"
        )
