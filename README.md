# 

This is a project for the Machine Learning course in the Master's program in Computer Engineering

### Acces
Click [here](https://iris-dataset-classification17.streamlit.app/) to access the app

### Dataset
To load the Iris dataset, use the following code:

```python
from sklearn.datasets import load_iris
iris = load_iris()

### Data exploration and model training and test
The source code where the dataset was explored, the data was prepared, and the model was developed can be found in the `Model` directory, in the file named `Proyect.py`. In the `Model`, there are also other file: `iris_model.pkl`, which are the pipeline and the best model files to be used in the app.

### APP
You will find two files: `requirements.txt` and `streamlit_app.py`. The first contains the libraries required to run the app. The second is the app code (model implementation and interface).