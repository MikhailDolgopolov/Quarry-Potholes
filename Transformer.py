import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Literal

OpType = Literal['', 'min', 'max', 'range', 'std', 'mean', 'var', 'sum', 'skew', 'kurt', 'median', 'iqr', 'ptp', 'cv']


class RollingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_params: Dict[str, List[OpType]], window_size: int = 5):
        self.window_size = window_size
        self.column_transform = column_params

    def fit(self, X: pd.DataFrame, y=None):
        missing_cols = [col for col in self.column_transform if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in X: {missing_cols}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame.")

        rolling_data = X.rolling(window=self.window_size, center=True)
        result = pd.DataFrame(index=X.index)

        for col, ops in self.column_transform.items():
            if col not in X.columns:
                continue

            col_rolling = rolling_data[col]
            agg_results = {}

            try:
                if any(op in ops for op in ['min', 'range', 'ptp', 'iqr']):
                    agg_results['min'] = col_rolling.min()
                if any(op in ops for op in ['max', 'range', 'ptp', 'iqr']):
                    agg_results['max'] = col_rolling.max()
                if 'std' in ops or 'cv' in ops:
                    agg_results['std'] = col_rolling.std()
                if 'mean' in ops or 'cv' in ops:
                    agg_results['mean'] = col_rolling.mean()
                if 'var' in ops:
                    agg_results['var'] = col_rolling.var()
                if 'sum' in ops:
                    agg_results['sum'] = col_rolling.sum()
                if 'skew' in ops:
                    agg_results['skew'] = col_rolling.skew()
                if 'kurt' in ops:
                    agg_results['kurt'] = col_rolling.kurt()
                if 'median' in ops:
                    agg_results['median'] = col_rolling.median()
                if 'iqr' in ops:
                    agg_results['q25'] = col_rolling.quantile(0.25)
                    agg_results['q75'] = col_rolling.quantile(0.75)

                for op in ops:
                    if op == '':
                        result[col] = X[col]
                    elif op == 'min':
                        result[f'{col}_min'] = agg_results['min']
                    elif op == 'max':
                        result[f'{col}_max'] = agg_results['max']
                    elif op == 'range' or op == 'ptp':
                        result[f'{col}_range'] = agg_results['max'] - agg_results['min']
                    elif op == 'std':
                        result[f'{col}_std'] = agg_results['std']
                    elif op == 'mean':
                        result[f'{col}_mean'] = agg_results['mean']
                    elif op == 'var':
                        result[f'{col}_var'] = agg_results['var']
                    elif op == 'sum':
                        result[f'{col}_sum'] = agg_results['sum']
                    elif op == 'skew':
                        result[f'{col}_skew'] = agg_results['skew']
                    elif op == 'kurt':
                        result[f'{col}_kurt'] = agg_results['kurt']
                    elif op == 'median':
                        result[f'{col}_median'] = agg_results['median']
                    elif op == 'iqr':
                        result[f'{col}_iqr'] = agg_results['q75'] - agg_results['q25']
                    elif op == 'cv':
                        result[f'{col}_cv'] = agg_results['std'] / agg_results['mean'].abs()

            except ZeroDivisionError:
                # print(f"Skipping cv for {col} due to division by zero.")
                continue
            except Exception as e:
                # print(f"Unexpected error in {col} with operation {op}: {e}")
                pass

        # Preserve columns
        preserve = ['hole', 'class']
        for output in preserve:
            if output in X.columns and output not in result.columns:
                result[output] = X[output]

        return result.ffill().bfill().dropna()
