import os
import mlflow
import mlflow.pyfunc

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://35.225.70.9:5000"
)

mlflow.set_tracking_uri(tracking_uri)


def test_model_loads():

    model = mlflow.pyfunc.load_model(
        "models:/oppe_mock_1@champion"
    )

    assert model is not None