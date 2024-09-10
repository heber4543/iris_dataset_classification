# librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

from sklearn.model_selection import StratifiedKFold, GridSearchCV
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

# best model
#  parámetros gridsearch
param_grid = {
    'C': [0.1, 1, 10, 100],  
    'penalty': ['l2'],  
    'solver': ['lbfgs', 'liblinear']
}

# gridsearch
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=stkfold)

# mejores parámetros y resultados
grid_search.fit(features, target)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Mejores parámetros: {best_params}")
print(f"Mejor puntuación: {best_score}")

# almacenamiento de métricas
accuracies = []
precisions = []
recalls = []
f1_scores = []
confusion_matrices = []

# evaluación con el mejor modelo
for train_index, test_index in stkfold.split(features, target):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]

    # entrenar modelo con mejores parámetros
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # predicción
    y_pred = best_model.predict(X_test)

    # métricas
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# matriz de confusión promedio
confusion_matrix_avg = np.mean(confusion_matrices, axis=0)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_avg, annot=True, fmt=".2f", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Average)')
plt.show()

# imprimir métricas promedio
print(f"Accuracy promedio: {np.mean(accuracies):.2f}")
print(f"Precision promedio: {np.mean(precisions):.2f}")
print(f"Recall promedio: {np.mean(recalls):.2f}")
print(f"F1-Score promedio: {np.mean(f1_scores):.2f}")

# importar el modelo
joblib.dump(best_model, "iris_model.plk")