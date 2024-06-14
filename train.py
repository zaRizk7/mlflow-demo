import mlflow
import mlflow.pyfunc
from mlflow import sklearn as mlflow_sklearn
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

# Example data
X1 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
X2 = np.array([[10], [20], [30]], dtype=np.float64)
y = np.array([15, 35, 55], dtype=np.float64)


# Create input structure
dataset = {
    "X1": X1,
    "X2": X2,
}

# Combine inputs for infer_signature
combined_input = [dataset["X1"], dataset["X2"]]

# Infer the signature
signature = infer_signature(dataset, y)


class Wrapper(RegressorMixin, BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, X1, X2, y):
        combined = np.hstack((X1, X2))
        self.model.fit(combined, y)
        return self

    def predict(self, X1, X2):
        combined = np.hstack((X1, X2))
        return self.model.predict(combined)


mlflow.set_tracking_uri("file:./mlruns")

# Log the model with MLflow
with mlflow.start_run():
    lr_mm = Wrapper(LinearRegression())
    lr_mm.fit(dataset["X1"], dataset["X2"], y)

    mlflow_sklearn.log_model(
        lr_mm,
        "model",
        signature=signature,
        input_example=dataset,
    )
