from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np
from filters.convolution import apply_convolution_filters 
from app.classification import classify_image 

app = FastAPI()

IMAGE_CLASSES = ["piedra", "papel", "tijera"]
LIMITATION_MESSAGE = f"El sistema solo está entrenado para reconocer imágenes de {IMAGE_CLASSES}"

def load_image_from_upload(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
        return img_np
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo cargar la imagen: {e}")

@app.post("/api/cnn/classify")
async def classify_image_endpoint(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=415, detail="Tipo de archivo no soportado. Sube JPEG o PNG.")
    
    image_bytes = await image.read()
    
    try:
        img_np = load_image_from_upload(image_bytes)

        filtered_img = apply_convolution_filters(img_np)

        prediction, confidence = classify_image(filtered_img, classes=IMAGE_CLASSES)
        
        return {
            "prediction": prediction,
            "confidence": f"{confidence:.4f}",
            "classes_supported": IMAGE_CLASSES,
            "limitations": LIMITATION_MESSAGE
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo interno en la clasificación: {e}")
    
#run using uvicorn cnn_service:app --reload --port 8002 at pipelineinteligentemlops/cnn_image