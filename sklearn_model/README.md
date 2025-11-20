# Modelo ML de Predicción (Scikit-Learn)
Puerto: ```8003```

# Propósito
Recibe archivos CSV, los procesa y utiliza un modelo de Random forest pre-entrenado (modelo_rf.pkl) para generar predicciones que luego son interpretadas por un LLM.

Endpoints Expuestos
```POST /model/linear/prediction```

# Ejemplo de Request/Response
Request:

Un csv (Binario)

Response (JSON):
```
{
  "result": "OK",
  "prediction": 3.7202500000000076,
  "explicacion": "La predicción de **3.72** en una escala de 0 a 100 para FIFA 16, a pesar de sus atributos positivos como un **Critic_Score de 82.00** y un **User_Score de 8.20**, y su origen de un Publisher y Developer consolidados (Electronic Arts, EA Canada), indica que el modelo predictivo ha asignado un potencial de ventas relativamente bajo.\n\nTécnicamente, este resultado sugiere que, para este modelo en particular, la combinación específica de características de entrada (Año de lanzamiento, Plataforma, Género, Puntuaciones, etc.) resultó en una salida en el extremo inferior de su rango de predicción. Podría deberse a:\n\n1.  **Calibración del Modelo:** La forma en que el modelo está escalado o calibrado para su salida. Un 3.72 podría ser la interpretación del modelo de un rendimiento bajo en su escala, incluso para un juego con buenas revisiones.\n2.  **Peso de las Características:** El modelo podría estar asignando un peso menor a las puntuaciones de crítica/usuario en comparación con otras características (como el año de lanzamiento en relación con el histórico de datos del modelo, o el propio género en un contexto determinado) para predecir las ventas en esta escala específica.\n3.  **Contexto Interno del Modelo:** Dentro de los patrones que el modelo ha aprendido de su conjunto de datos de entrenamiento, esta combinación particular de atributos no se correlaciona con un valor alto en la escala de 0 a 100 para las ventas.\n\nEn resumen, el modelo no ha encontrado en este conjunto de características el patrón para un rendimiento de ventas superior en su escala de predicción, independientemente de la calidad percibida del juego por críticos y usuarios."
}
```
# Ejecución Local

```uvicorn app.ml_service:app --host 0.0.0.0 --port 8003 --reload```