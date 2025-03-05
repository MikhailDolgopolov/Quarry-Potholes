import numpy as np

import re

import pandas as pd


def calculate_summed_magnitude(df, prefix):
    # Select columns starting with the specified prefix
    pattern = f"{prefix}.*[XYZ]$"
    cols = [col for col in df.columns if re.match(pattern, col)]
    if len(cols) != 3:
        raise ValueError(f"Expected exactly 3 columns with prefix '{prefix}', but found {len(cols)}: {cols}")

    # Calculate the magnitude row-wise
    magnitude = np.sqrt((df[cols] ** 2).sum(axis=1))
    return magnitude


def convert_dash_to_nan(df):
    """
    Convert columns with '-' as missing values to numeric type in a pandas DataFrame.
    Only columns where all non-'-' values (after stripping whitespace) are numeric are converted.

    Parameters:
        df (pandas.DataFrame): Input DataFrame

    Returns:
        pandas.DataFrame: DataFrame with corrected numeric columns
    """
    for col in df.columns:
        # Check if the column is of object dtype (typically strings in pandas)
        if df[col].dtype == 'object':
            # Select values where, after stripping whitespace, the value is not '-'
            mask = df[col].str.strip() != '-'
            non_dash = df[col][mask]
            # Attempt to convert these non-'-' values to numeric, coercing errors to NaN
            converted = pd.to_numeric(non_dash, errors='coerce')
            # If all converted values are not NaN, the column is likely numeric with '-' as missing
            if converted.notna().all():
                # Replace values that strip to '-' with NaN
                df[col] = df[col].apply(lambda x: np.nan if x.strip() == '-' else x)
                # Convert the entire column to numeric
                df[col] = pd.to_numeric(df[col])
    return df