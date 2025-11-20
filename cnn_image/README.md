# CNN para clasificaicón de imagenes
Puerto: ```8002```

# Propósito
Clasifica imágenes subidas por el usuario utilizando una Red Neuronal Convolucional (CNN) igualmente esta predicción es analisada por un LLM.

# Endpoints Expuestos
```POST /api/cnn/classify```

# Ejemplo de Request/Response
Request:

Una imagen (.jpg, .png, .jpeg) 

Response (JSON):
```
{
"prediction": "tijera",
"confidence": 0.921,
"explanation": "Como analista experto en imagenes, interpreto el resultado de la siguiente manera:\n\nEl modelo de clasificacion de imagenes ha predicho que el objeto detectado en la imagen es
una ** \"tijera\" **. Esta predicción se acompaña de una ** confianza del 0.92 **. \n\n ** Explicacion tecnica :** \n\nLa confianza de 0.92 (o 92%) indica que el modelo asigno una alta probabilidad a la
clase \"tijera\" como la más representativa para la entrada visual analizada. Es una metrica interna que refleja la certidumbre del modelo basada en los patrones aprendidos durante su entrenamie
nto. Un valor tan elevado sugiere una fuerte correlacion entre las caracteristicas extraidas de la imagen y los rasgos distintivos que el modelo asocia con el concepto de \"tijera\".",
"classes_supported": [
"piedra",
"papel",
"tijera"
]
"limitations": "El sistema solo esta entrenado para reconocer imagenes de gestos de 'piedra', 'papel' y 'tijera' en condiciones similares al dataset de entrenamiento. Puede fallar
istintos, imágenes muy oscuras/borrosas o ángulos poco comunes."
}

```
# Ejecución Local

```uvicorn app.cnn_rps_service:app --host 0.0.0.0 --port 8002 --reload```