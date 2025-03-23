import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Literal

# Define allowed operation types
OpType = Literal['', 'min', 'max', 'range', 'std', 'mean', 'var', 'sum']

class MultiWindowRollingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies rolling window operations across multiple window sizes
    to specified columns and computes various statistics.

    Parameters
    ----------
    column_params : Dict[str, List[OpType]]
        A dictionary mapping column names to lists of operations to compute.
        Operations can be: '', 'min', 'max', 'range', 'std', 'mean', 'var', 'sum'.
        The empty string indicates that the original column should be kept.
    window_sizes : List[int]
        A list of window sizes to apply the rolling operations.
    preserve_cols : List[str], default=['hole', 'class']
        Columns to preserve in the output DataFrame.
    """

    def __init__(self, column_params: Dict[str, List[OpType]], window_sizes: List[int], preserve_cols=None):
        if preserve_cols is None:
            preserve_cols = ['hole', 'class']
        self.column_params = column_params
        self.window_sizes = window_sizes
        self.preserve_cols = preserve_cols

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
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        missing_cols = [col for col in self.column_params if col not in X.columns]
        if missing_cols:
            raise ValueError(f"The following columns were not found in X: {missing_cols}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Apply the rolling window transformation for multiple window sizes.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : None
            Ignored.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with computed rolling window features for each window size.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        # Initialize result DataFrame with the original index
        result = pd.DataFrame(index=X.index)

        # Copy original columns where '' is specified
        for col, ops in self.column_params.items():
            if '' in ops and col in X.columns:
                result[col] = X[col]

        # Process each window size
        for ws in self.window_sizes:
            for col, ops in self.column_params.items():
                if col not in X.columns:
                    continue

                # Identify basic operations and required aggregates
                basic_ops = [op for op in ops if op in ['min', 'max', 'std', 'mean', 'var', 'sum']]
                required_aggs = set(basic_ops)
                if 'range' in ops:
                    required_aggs.update(['min', 'max'])

                if required_aggs:
                    # Compute rolling aggregates efficiently
                    rolling = X[col].rolling(window=ws, center=True)
                    agg_df = rolling.agg(list(required_aggs))

                    # Add basic operations to result
                    for op in basic_ops:
                        result[f'{col}_{op}_ws{ws}'] = agg_df[op]

                    # Compute and add derived operations
                    if 'range' in ops:
                        result[f'{col}_range_ws{ws}'] = agg_df['max'] - agg_df['min']

        # Preserve specified columns
        for col in self.preserve_cols:
            if col in X.columns and col not in result.columns:
                result[col] = X[col]

        # Handle missing values
        result = result.ffill().bfill().dropna()
        return result