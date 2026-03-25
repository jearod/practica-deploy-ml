"""
Aplicación FastAPI para clasificacion de flores con 4 caracteristicas de entrada.

Esta API permite clasificar flores en 1 de 3 diferentes especies

Características principales:
- Endpoint /: Informacion de la API, nombre, version, algoritmo usado, ejemplo de llamada
- Endpoint /predict: Recibe las 4 features de Iris en JSON y devuelve la especie predicha
- Endpoint /health: Devuelve `{"status": "healthy"}`

El modelo utilizado es un clasificador lineal de Regression Logistica entrenado previamente y guardado en 'model.pkl'.
"""

import joblib
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# ------------------------- Cargar modelo entrenado -------------------------
# El modelo de Machine Learning se carga al iniciar la aplicación.
# Se utiliza joblib para deserializar el modelo guardado en un archivo pickle.
# Si no se puede cargar, se lanza un error crítico que detiene la aplicación.
CLASSES = ["setosa", "versicolor", "virginica"]
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Modelo cargado exitosamente.")
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")


# ------------------------- Endpoint raiz -------------------------
# Endpoint GET para obtener información general sobre la API.
#
# Método: GET
# Ruta: /
# Sin parámetros.
#
# Retorna metadatos de la API, incluyendo nombre, descripción, versión y lista de características disponibles.
# Útil para que los clientes conozcan las capacidades del servicio.
#
# Respuesta (200):
# {
#   "service": "Flower Classifier ML API",
#   "description": "API de ejemplo para Clasificar flores con FastAPI y ML",
#   "version": "1.0",
#   "algorithm":  "Logistic Regression",
#   "features": ["predict", "health"]
# }
@app.route("/")
def home():
    return {
        "service": "Flower Classifier ML API",
        "description": "API de ejemplo para Clasificar flores con FastAPI y ML",
        "version": "1.0",
        "algorithm":  "Logistic Regression",
        "features": ["predict", "health"]
    }


# ------------------------- Endpoint predict -------------------------
# Endpoint POST para hacer una predicción individual de retraso en un vuelo.
#
# Método: POST
# Ruta: /predict
# Cuerpo de la solicitud: JSON con sepal_length ,sepal_width, petal_length y petal_width
#
# Proceso:
# 1. Pydantic valida automáticamente los datos según el modelo irisData.
# 2. Usa el modelo para clasificar la flor
# 4. Determina si la flor es setosa, versiclor y virginica.
# 5. Incrementa el contador de predicciones.
# 6. Retorna un JSON con prediction, prediction_index, probabilities,
#
# Respuesta exitosa (200):
#{
#    "prediction": "setosa",
#    "prediction_index": 0,
#    "probabilities": {
#        "setosa": 0.97,
#        "versicolor": 0.02,
#        "virginica": 0.01
#    },
#    "confidence": 0.97,
#    "status": "success"
#}
#
# Errores posibles:
# - 422: Datos inválidos (Pydantic validation error).
# - 500: Error interno (problema con el modelo o procesamiento).
@app.route('/predict', methods=['POST'])
def predict():    
    try:
        # Obtener datos del request
        data = request.get_json()
        input_data = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        features = scaler.transform(input_data)

        # Realizar predicción
        probabilities = model.predict_proba(features)[0]
        # Determinar si se considera retrasado basado en umbral 0.5
        prediction_index = int(np.argmax(probabilities))
        prediction_name = CLASSES[prediction_index]
        confidence = float(probabilities[prediction_index])

         # Retornar respuesta con resultados
        return jsonify({ 
            "prediction": prediction_name,
            "prediction_index": prediction_index,
            "probabilities": {
                "setosa": round(float(probabilities[0]), 2),
                "versicolor": round(float(probabilities[1]), 2),
                "virginica": round(float(probabilities[2]), 2)
            },
            "confidence": round(confidence, 2),
            "status": "success"
         })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
    
# ------------------------- Endpoint health -------------------------
# Endpoint GET para comprobar que el servicio esta arriba de la API.
#
# Método: GET
# Ruta: /health
# Sin parámetros.
#
# Retorna:
# - status: Estado del servicio.
#
# Útil para monitoreo y debugging del servicio.
#
# Respuesta (ok):
# {
#   "status": "healthy"
# }
@app.route('/health')
def health():
    # Retornar status healthy
    return {
        "status": "healthy"
    }


if __name__ == '__main__':
    # Render/Railway asigna el puerto mediante la variable PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)