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

# indicaciones
st.markdown("### This interface helps implement the linear regression model to predict the flower species. The model was trained with the Iris dataset. Whenever you are asked for information, you will be informed of the minimum and maximum values you can enter.")

# entradas de usuario

# sepal length
st.markdown("#### Sepal length (cm). Minimum value: 4.30 / Maximum value: 7.90")
sepal_length = st.number_input("Sepal length (cm)", min_value=0.0)
# sepal width 
st.markdown("#### Sepal width  (cm). Minimum value: 2.00 / Maximum value: 4.40")
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0)
# petal length 
st.markdown("#### Petal length  (cm). Minimum value: 1.00 / Maximum value: 6.90")
petal_length = st.number_input("Petal length (cm)", min_value=0.0)
# petal width 
st.markdown("#### Petal width  (cm). Minimum value: 0.10 / Maximum value: 2.50")
petal_width = st.number_input("Petal width (cm)", min_value=0.0)

# predicción
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]
    st.write(f"The flower is: **{pred_class}**")