"""
Module: preprocess
Contains helper functions for preprocessing the House Prices dataset.
"""

import pandas as pd


def fill_missing(df: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """
    Fill missing values in a dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        numeric_cols (list[str]): List of continuous columns.
        cat_cols (list[str]): List of categorical columns.

    Returns:
        pd.DataFrame: Dataframe with missing values filled.
    """
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df
