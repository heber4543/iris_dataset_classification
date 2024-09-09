import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# cargar dataset
iris = load_iris()

# cargar el modelo desde el archivo
model = joblib.load('Model/iris_model.pkl')

# titulo aplicación
st.title("Iris flower classifier with logistic regression")

# entradas de usuario
sepal_length = st.number_input("Sepal length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0)
petal_length = st.number_input("Petal length (cm)", min_value=0.0)
petal_width = st.number_input("Petal width (cm)", min_value=0.0)

# predicción
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]
    st.write(f"The flower is: **{pred_class}**")
