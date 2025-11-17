from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import pandas as pd
from google import genai
from dotenv import load_dotenv
import os
import io

app = FastAPI()

load_dotenv()
gemini_api_key = os.getenv("gemini_api_key")
model_path = os.getenv("model_path")

client = genai.Client(api_key=gemini_api_key)

@app.post("/model/linear/prediction")
async def make_prediction(csv: UploadFile = File(...)):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return {"result": "Error", "message": f"No se encontró el archivo del modelo en la ruta: {model_path}"}
    except Exception as e:
        return {"result": "Error", "message": f"Error al cargar el modelo {e}"}
    content = await csv.read()

    csv_stream = io.String(content.decode('utf-8'))

    df = pd.read_csv(csv_stream, sep=",")

    if df.empty:
        return {"result": "Error", "message": "El archivo CSV está vacío."}
    prep_data = df.iloc[0].to_frame().T 

    try:
        predicted_value = model.predict(prep_data)[0]
        datos_string = prep_data.iloc[0].to_dict()
        datos_str_limpio = ", ".join([f"{k}: {v:.2f}" for k, v in datos_string.items()])

    except Exception as e:
        return {"result": "Error", "message": f"Error durante la predicción con el modelo: {e}"}
    
    try:
        prompt_explicacion = (
            f"""Eres un analista de ventas experto. Basado en las siguientes características de ventas de videojuegos:
            {datos_str_limpio} se predijo {predicted_value:.2f} en una escala de 0 a 100. Proporciona una explicación clara, breve y técnica de este resultado.
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
                "result" : "OK",
                "prediction" : predicted_value,
                "explicacion" : explicacion.strip()
            }
        else:
            return {"result": "Error", "message": "El modelo de IA no devolvió contenido."}
    except HTTPException as e:
        raise e
    except Exception as e:
        return {"result": "Error", "message" : f"Fallo interno en la predicción: {e}"}
    
#run using uvicorn model_service:app --reload --port 8003 at pipelineinteligentemlops/sklearn_model