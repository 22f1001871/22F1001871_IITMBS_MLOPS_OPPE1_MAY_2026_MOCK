import os
import mlflow.pyfunc
import pandas as pd

from sklearn.metrics import accuracy_score

import mlflow

mlflow.set_tracking_uri("http://34.70.155.232:5000")


def test_model_prediction():

    model = mlflow.pyfunc.load_model(
        "models:/oppe_mock_1@champion"
    )

    iris_v0 = pd.read_csv("data/iris_v0.csv")
    iris_v1 = pd.read_csv("data/iris_v1.csv")

    df = pd.concat(
        [iris_v0, iris_v1],
        ignore_index=True
    )

    X = df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    ]

    y = df["species"]

    predictions = model.predict(X)

    assert len(predictions) == len(y)

    accuracy = accuracy_score(y, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    assert accuracy >= 0.80