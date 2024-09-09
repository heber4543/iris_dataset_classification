import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# cargar modelo
iris = load_iris()

# Cargar el modelo desde el archivo
model = joblib.load('Model/iris_model.pkl')

# Título de la aplicación
st.title("Predicción de Flores Iris")

# Entradas de usuario
sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0)
sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0)
petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0)
petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0)

# Realizar predicción al hacer clic en el botón
if st.button("Predecir"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]
    st.write(f"La flor es probablemente: **{pred_class}**")
