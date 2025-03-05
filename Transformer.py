import random

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Dict, List, Literal


class RollDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_params: Dict[str, List[Literal['', 'min', 'max', 'range', 'std', 'mean']]], window_size=5):
        self.window_size = window_size
        self.column_transform = column_params

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rolling_data = X.rolling(window=self.window_size, center=True)
        result = pd.DataFrame()

        # Process each column with its parameters
        for col, params in self.column_transform.items():
            if col not in X.columns:
                continue
            for param in params:
                if param == 'min':
                    result[f'{col}_min'] = rolling_data[col].min()
                elif param == 'max':
                    result[f'{col}_max'] = rolling_data[col].max()
                elif param == 'range':
                    result[f'{col}_range'] = rolling_data[col].max() - rolling_data[col].min()
                elif param == 'std':
                    result[f'{col}_std'] = rolling_data[col].std()
                elif param == 'mean':
                    result[f'{col}_mean'] = rolling_data[col].mean()
                elif param == '':
                    result[col] = X[col]  # Directly copy the column

        # Explicitly preserve the 'hole' column
        if 'hole' in X.columns:
            result['hole'] = X['hole']

        result = result.ffill().bfill()
        result = result.dropna()
        return result

    def roll_data(self, df: pd.DataFrame):
        result = self.transform(df)
        try:
            # Keep lat/lon smoothing if needed
            result['lat'] = df['lat'].rolling(window=self.window_size, center=True).mean()
            result['lon'] = df['lon'].rolling(window=self.window_size, center=True).mean()
        except Exception as e:
            print(e)

        result = result.ffill().bfill()
        result = result.dropna()
        return result


