import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd

# Load the model in `python_function` format
model = mlflow_sklearn.load_model("mlruns/0/b0f7ac4ad59c4657b3632c9aba584ff9/artifacts/model")

# Example data
X1 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
X2 = np.array([[10], [20], [30]], dtype=np.float64)
y = np.array([15, 35, 55], dtype=np.float64)

print(model.predict(X1, X2))