"""
Module: train
Responsible for training and persisting the ML model for the House Prices dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
import joblib
from pathlib import Path


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Logarithmic Error (RMSLE)."""
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(y_pred, 0)))


def build_model(data: pd.DataFrame) -> dict[str, float]:
    """
    Build, train, and evaluate a Ridge Regression model on the dataset.

    Returns a dictionary containing the RMSLE score.
    """

    features_cont = ["GrLivArea", "GarageArea"]
    features_cat = ["MSZoning", "HouseStyle"]
    target = "SalePrice"

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    X_train = train_df[features_cont + features_cat]
    y_train = train_df[target]
    X_test = test_df[features_cont + features_cat]
    y_test = test_df[target]

    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train[features_cont])
    X_test_cont = scaler.transform(X_test[features_cont])

    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    X_train_cat = encoder.fit_transform(X_train[features_cat])
    X_test_cat = encoder.transform(X_test[features_cat])

    X_train_proc = np.hstack([X_train_cont, X_train_cat])
    X_test_proc = np.hstack([X_test_cont, X_test_cat])

    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)
    rmsle = compute_rmsle(y_test, y_pred)

    Path("../models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "../models/model.joblib")
    joblib.dump(scaler, "../models/scaler.joblib")
    joblib.dump(encoder, "../models/encoder.joblib")

    return {"rmsle": round(rmsle, 5)}
