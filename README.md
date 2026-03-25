# 🌸 Iris Flower Classifier API

Este proyecto es una **API REST** de alto rendimiento desarrollada con **Flask**. Su objetivo es clasificar especies de flores Iris (Setosa, Versicolor, Virginica) basándose en cuatro características físicas, utilizando un modelo de Machine Learning previamente entrenado.



---

## 📖 Descripción del Proyecto

El sistema automatiza el flujo de trabajo de Ciencia de Datos, desde el preprocesamiento hasta el despliegue en producción. 

### El Modelo
Se utiliza un algoritmo de **Regresión Logística**, ideal por su balance entre simplicidad y eficacia en problemas linealmente separables. El modelo transforma las entradas mediante la función sigmoide:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Donde $z$ es la combinación lineal de las características de la flor.



---

## 🛠️ Estructura del Repositorio

* **`train_model.py`**: Script de entrenamiento. Carga el dataset, aplica `StandardScaler`, entrena el modelo y exporta los archivos `.pkl`.
* **`app.py`**: El núcleo de la API Flask. Maneja la lógica de predicción, escalado en tiempo real y manejo de errores.
* **`model.pkl`**: Pesos del modelo entrenado.
* **`scaler.pkl`**: Parámetros de normalización (media y desviación estándar) necesarios para mantener la integridad de los datos de entrada.
* **`requirements.txt`**: Dependencias del proyecto.

---

## 🚀 Instalación y Uso rápido

### 1. Preparar el entorno
Se recomienda usar un entorno virtual para evitar conflictos de dependencias:

```bash
# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar el servidor

Una vez activado el entorno y asegurándote de que los archivos `.pkl` existen, lanza la aplicación:

Bash

```
python app.py
```

La API se levantará por defecto en `http://localhost:5000`.

----------

## 🛣️ Endpoints Disponibles


### Endpoints Disponibles

| Método | Ruta | Funcionalidad |
| :--- | :--- | :--- |
| **GET** | `/` | Metadatos de la API (nombre, versión, algoritmo). |
| **GET** | `/health` | Verificación de estado del servicio (Health Check). |
| **POST** | `/predict` | Envío de datos técnicos para clasificación inmediata. |

### Ejemplo de Petición (POST `/predict`)

**Cuerpo de la solicitud (JSON):**

JSON

```
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

**Respuesta Exitosa (JSON):**

JSON

```
{
    "prediction": "setosa",
    "prediction_index": 0,
    "confidence": 0.97,
    "probabilities": {
        "setosa": 0.97,
        "versicolor": 0.02,
        "virginica": 0.01
    },
    "status": "success"
}
```
----------

## 📊 Tecnologías y Herramientas

-   **Lenguaje:** Python 3.x    
-   **Framework Web:** Flask  
-   **ML & Data:** Scikit-Learn, NumPy, Pandas.  
-   **Serialización:** Joblib.
    
----------

> **⚠️ Nota Crítica:** Este proyecto utiliza un escalador (`StandardScaler`). Es fundamental que cualquier entrada nueva pase por el método `scaler.transform()` antes de ser procesada por el modelo. Ignorar este paso resultará en predicciones incorrectas debido a la diferencia de escalas.