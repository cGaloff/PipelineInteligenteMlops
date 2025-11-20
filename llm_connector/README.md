# Chat LLM Service
Puerto: ```8000```

# Propósito
Procesa el historial de conversación y genera respuestas utilizando un modelo de lenguaje (LLM).

# Endpoints Expuestos
```POST /chat```

# Ejemplo de Request/Response
Request (JSON):
```
{
  "messages": [
    {"role": "user", "content": "Hola, que país"},
  ]
}
```
Response (JSON):

```
{
  "response": "Rusia y Canadá son países más grandes que China. Rusia es el país más grande del mundo con aproximadamente 17 millones de km², mientras que Canadá es el segundo más grande con cerca de 10 millones de km². "
}
```

# Ejecución Local
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload