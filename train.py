import os
import joblib
import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

####################################################
# LABEL ENCODING
####################################################

LABEL_MAP = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}


####################################################
# CUSTOM IMPUTATION
####################################################

def impute_last10_same_species(df):

    df = df.copy()

    feature_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    for feature in feature_cols:

        for idx in df.index:

            if pd.isna(df.loc[idx, feature]):

                species = df.loc[idx, "species"]

                previous_rows = df.loc[:idx-1]

                previous_rows = previous_rows[
                    previous_rows["species"] == species
                ]

                values = (
                    previous_rows[feature]
                    .dropna()
                    .tail(10)
                )

                if len(values) > 0:
                    df.loc[idx, feature] = values.mean()

                else:
                    # fallback if no previous samples exist
                    df.loc[idx, feature] = df[
                        df["species"] == species
                    ][feature].mean()

    return df


####################################################
# TRAINING FUNCTION
####################################################

def train_model(df, iteration):

    print(f"\n========== ITERATION {iteration} ==========\n")

    df = impute_last10_same_species(df)

    df["target"] = df["species"].map(LABEL_MAP)

    X = df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    ]

    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_val,
            predictions
        )
    )

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        f"models/model_iteration_{iteration}.joblib"
    )

    return model


####################################################
# MAIN
####################################################

if __name__ == "__main__":

    iris_v0 = pd.read_csv("data/iris_v0.csv")

    iris_v1 = pd.read_csv("data/iris_v1.csv")

    ################################################
    # ITERATION 1
    ################################################

    train_model(
        iris_v0,
        iteration=1
    )

    ################################################
    # ITERATION 2
    ################################################

    merged = pd.concat(
        [iris_v0, iris_v1],
        ignore_index=True
    )

    train_model(
        merged,
        iteration=2
    )