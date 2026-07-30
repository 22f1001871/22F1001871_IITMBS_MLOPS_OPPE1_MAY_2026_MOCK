import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


############################################################
# MLFLOW
############################################################

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://35.225.70.9:5000"
)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("OPPE_MOCK_1")

client = MlflowClient()

############################################################
# HYPERPARAMETERS
############################################################

depths = [3, 5, 7, None]
criterions = ["gini", "entropy"]

############################################################
# GLOBAL BEST MODEL
############################################################

best_accuracy = 0
best_model = None
best_version = None


############################################################
# TRAINING FUNCTION
############################################################

def train_model(df, iteration):

    global best_accuracy
    global best_model
    global best_version

    print(f"\n========== ITERATION {iteration} ==========\n")

    X = df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    ]

    y = df["species"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    os.makedirs("models", exist_ok=True)

    for depth in depths:

        for criterion in criterions:

            run_name = f"Iteration_{iteration}_depth_{depth}_{criterion}"

            with mlflow.start_run(run_name=run_name):

                model = DecisionTreeClassifier(
                    max_depth=depth,
                    criterion=criterion,
                    random_state=42,
                )

                model.fit(X_train, y_train)

                predictions = model.predict(X_val)

                accuracy = accuracy_score(
                    y_val,
                    predictions,
                )

                report = classification_report(
                    y_val,
                    predictions,
                )

                print("=" * 60)
                print(run_name)
                print(f"Accuracy : {accuracy:.4f}")
                print(report)

                ################################################
                # PARAMETERS
                ################################################

                mlflow.log_param("iteration", iteration)
                mlflow.log_param("max_depth", depth)
                mlflow.log_param("criterion", criterion)

                ################################################
                # METRICS
                ################################################

                mlflow.log_metric("accuracy", accuracy)

                ################################################
                # REGISTER MODEL
                ################################################

                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name="oppe_mock_1",
                )

                ################################################
                # BEST MODEL
                ################################################

                if accuracy > best_accuracy:

                    best_accuracy = accuracy
                    best_model = model
                    best_version = model_info.registered_model_version

    print(f"\nBest Accuracy So Far : {best_accuracy:.4f}")


############################################################
# MAIN
############################################################

if __name__ == "__main__":

    iris_v0 = pd.read_csv("data/iris_v0.csv")
    iris_v1 = pd.read_csv("data/iris_v1.csv")

    ########################################################
    # ITERATION 1
    ########################################################

    train_model(
        iris_v0,
        iteration=1,
    )

    ########################################################
    # ITERATION 2
    ########################################################

    merged = pd.concat(
        [iris_v0, iris_v1],
        ignore_index=True,
    )

    train_model(
        merged,
        iteration=2,
    )

    ########################################################
    # SAVE BEST MODEL
    ########################################################

    joblib.dump(
        best_model,
        "models/best_model.joblib",
    )

    ########################################################
    # SET CHAMPION ALIAS
    ########################################################

    client.set_registered_model_alias(
        name="oppe_mock_1",
        alias="champion",
        version=best_version,
    )

    print("\n" + "=" * 60)
    print(f"BEST ACCURACY : {best_accuracy:.4f}")
    print(f"CHAMPION VERSION : {best_version}")
    print("=" * 60)