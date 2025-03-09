import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Literal

# Allowed operations for each column.
OpType = Literal['', 'min', 'max', 'range', 'std', 'mean']


class RollingWindowTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a rolling window to specified columns and computes
    various statistics (min, max, range, std, mean) for each window.

    Parameters
    ----------
    column_params : dict
        A dictionary mapping column names to a list of operations to compute.
        Operations can be one of: '', 'min', 'max', 'range', 'std', 'mean'.
        The empty string indicates that the original column should be kept.
    window_size : int, default=5
        The size of the rolling window.
    """

    def __init__(self, column_params: Dict[str, List[OpType]], window_size: int = 5):
        self.window_size = window_size
        self.column_transform = column_params

    def fit(self, X: pd.DataFrame, y=None):
        """
        Validate that all required columns exist in X.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : None
            Ignored.

        Returns
        -------
        self
        """
        missing_cols = [col for col in self.column_transform if col not in X.columns]
        if missing_cols:
            raise ValueError(f"The following columns were not found in X: {missing_cols}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Apply the rolling window transformation to the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : None
            Ignored.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with computed rolling window features and filled missing values.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        # Initialize rolling object and an empty result DataFrame using the original index.
        rolling_data = X.rolling(window=self.window_size, center=True)
        result = pd.DataFrame(index=X.index)

        # Process each column as per the specified operations.
        for col, ops in self.column_transform.items():
            if col not in X.columns:
                continue

            # Compute rolling aggregates once per column if needed.
            col_rolling = rolling_data[col]
            agg_results = {}
            if any(op in ops for op in ['min', 'range']):
                agg_results['min'] = col_rolling.min()
            if any(op in ops for op in ['max', 'range']):
                agg_results['max'] = col_rolling.max()
            if 'std' in ops:
                agg_results['std'] = col_rolling.std()
            if 'mean' in ops:
                agg_results['mean'] = col_rolling.mean()

            # Map each operation to its resulting column in the output.
            for op in ops:
                if op == 'min':
                    result[f'{col}_min'] = agg_results['min']
                elif op == 'max':
                    result[f'{col}_max'] = agg_results['max']
                elif op == 'range':
                    # Ensure min and max are computed.
                    result[f'{col}_range'] = agg_results['max'] - agg_results['min']
                elif op == 'std':
                    result[f'{col}_std'] = agg_results['std']
                elif op == 'mean':
                    result[f'{col}_mean'] = agg_results['mean']
                elif op == '':
                    # Copy the original column.
                    result[col] = X[col]
                else:
                    raise ValueError(f"Operation '{op}' not recognized for column '{col}'.")

        # Optionally preserve additional columns (example: 'hole').
        if 'hole' in X.columns and 'hole' not in result.columns:
            result['hole'] = X['hole']

        # Fill missing values using forward-fill then backward-fill and drop any remaining NaNs.
        result = result.ffill().bfill().dropna()
        return result

    def roll_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        An alternative method that applies the transform and additionally
        smooths 'lat' and 'lon' columns if they exist.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The transformed data with 'lat' and 'lon' smoothed.
        """
        result = self.transform(X)
        # Smooth latitude and longitude if available.
        for coord in ['lat', 'lon']:
            if coord in X.columns:
                result[coord] = X[coord].rolling(window=self.window_size, center=True).mean()
        result = result.ffill().bfill().dropna()
        return result
