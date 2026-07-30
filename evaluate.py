import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

############################################################
# MLFLOW
############################################################

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://35.225.70.9:5000"
)

mlflow.set_tracking_uri(tracking_uri)

############################################################
# LOAD CHAMPION MODEL
############################################################

model = mlflow.sklearn.load_model(
    "models:/oppe_mock_1@champion"
)

############################################################
# LOAD LATEST DATA
############################################################

iris_v0 = pd.read_csv("data/iris_v0.csv")
iris_v1 = pd.read_csv("data/iris_v1.csv")

merged = pd.concat(
    [iris_v0, iris_v1],
    ignore_index=True
)

############################################################
# TEST SPLIT
############################################################

X = merged[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

y = merged["species"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

############################################################
# EVALUATE
############################################################

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy : {accuracy:.4f}")

print(classification_report(
    y_test,
    predictions
))