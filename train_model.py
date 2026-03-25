# ============================================================
# PASO 1: Entrenar y guardar el modelo
# Archivo: train_model.py
# ============================================================

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target


# -------------------------------------------------------
# TODO 1: Divide los datos en entrenamiento y prueba
# -------------------------------------------------------

# Tu código aquí:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# -------------------------------------------------------
# TODO 2: Elige y entrena un algoritmo de clasificación
# -------------------------------------------------------

# Tu código aquí:
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------------------------------
# TODO 3: Evalúa el modelo y muestra métricas
# -------------------------------------------------------

# Tu código aquí:

print(classification_report(y_test, model.predict(X_test_scaled)))
# -------------------------------------------------------
# TODO 4: Guarda el modelo como 'model.pkl'
# -------------------------------------------------------

# Tu código aquí:
joblib.dump(model, "model.pkl")
print('¡Modelo guardado correctamente!')