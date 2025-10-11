"""
Module: inference
Used for making predictions using the persisted House Prices model.
"""

import pandas as pd
import numpy as np
import joblib


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Perform inference using the persisted model and preprocessing objects.
    """
    features_cont = ["GrLivArea", "GarageArea"]
    features_cat = ["MSZoning", "HouseStyle"]

    model = joblib.load("../models/model.joblib")
    scaler = joblib.load("../models/scaler.joblib")
    encoder = joblib.load("../models/encoder.joblib")

    for col in features_cont:
        input_data[col] = input_data[col].fillna(input_data[col].median())
    for col in features_cat:
        input_data[col] = input_data[col].fillna(input_data[col].mode()[0])

    X_cont = scaler.transform(input_data[features_cont])
    X_cat = encoder.transform(input_data[features_cat])
    X_proc = np.hstack([X_cont, X_cat])

    predictions = model.predict(X_proc)
    return np.maximum(predictions, 0)
