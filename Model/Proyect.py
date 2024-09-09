# librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import joblib

# cargar dataset
iris = load_iris()

# variables de target y características
features = iris.data
target = iris.target

# crear instancia de kfold y de LR
stkfold = StratifiedKFold(n_splits=5) 
model = LogisticRegression(max_iter=200)  

# almacenamiento de métricas
accuracies = []
precisions = []
recalls = []
f1_scores = []
confusion_matrices = []

# ciclo para divir los datos en folds y entrenar modelo
for train_index, test_index in stkfold.split(features, target):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    # entrenar modelo
    model.fit(X_train, y_train)
    
    # predicción
    y_pred = model.predict(X_test)
    
    # métricas
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# imprimir métricas por partición
print(f"Precisión (Accuracy) por partición: {accuracies}")
print(f"Precisión (Precision) por partición: {precisions}")
print(f"Sensibilidad (Recall) por partición: {recalls}")
print(f"F1-Score por partición: {f1_scores}")

# calcular matriz de confusión promedio
confusion_matrix_avg = np.mean(confusion_matrices, axis=0)

# garfica matriz de confusión promedio
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_avg, annot=True, fmt=".2f", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Average)')
plt.show()

# imprimir las métricas promedio
print(f"Precisión (Accuracy) promedio: {np.mean(accuracies)}")
print(f"Precisión (Precision) promedio: {np.mean(precisions)}")
print(f"Sensibilidad (Recall) promedio: {np.mean(recalls)}")
print(f"F1-Score promedio: {np.mean(f1_scores)}")

# guardar el modelo
joblib.dump(model, "/workspaces/iris_dataset_classification/Model/iris_model.pkl")