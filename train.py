# train.py
"""
Training script for Healthy Life Insurance charge prediction model.

- Loads insurance.csv
- Builds preprocessing + regression pipeline
- Trains the model
- Saves the trained pipeline as model.joblib
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def load_data(csv_path: str = "insurance.csv"):
    """Load the insurance dataset."""
    df = pd.read_csv(csv_path)
    X = df.drop("charges", axis=1)
    y = df["charges"]
    return X, y


def build_pipeline():
    """Build preprocessing + model pipeline."""
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_and_save_model(
    csv_path: str = "/content/sample_data/insurance.csv",
    model_path: str = "model.joblib",
):
    """Train the pipeline on the data and save it to disk."""
    X, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:,.2f}")
    print(f"Test R²:  {r2:.3f}")

    # Save the trained pipeline
    joblib.dump(pipeline, model_path)
    print(f"Saved trained model to {model_path}")

    return pipeline


if __name__ == "__main__":
    train_and_save_model()
