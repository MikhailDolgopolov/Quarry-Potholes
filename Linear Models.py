import os
import sys
from contextlib import contextmanager

import numpy as np
from pygam import LogisticGAM, PoissonGAM
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygam")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

@contextmanager
def suppress_stdout():
    """Temporarily redirect stdout to null device."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
from data_read import load_prepared

ws=10
filename = f"models/linear_{ws}.pkl"

data_file = f"data/class{ws}"
big_df = load_prepared(data_file, sample_frac=0.4)

# Split data.
train_df, test_df = train_test_split(big_df, test_size=0.2)



# if (y_train <= 0).any():
#     print("Warning: Target contains zeros or negatives. Adding 1 to all values.")
#     y_train = y_train + 1
#     y_test = y_test + 1
# Define parameter grid
param_grid = {
    'lam': [0.5, 1, 5, 10],
    'n_splines': [10, 15]
}
print(param_grid)
X_train, y_train = train_df.drop(columns=['class']), train_df['class']
X_test, y_test = test_df.drop(columns='class'), test_df['class']

best_score = float('inf')
best_model = None

# Manual grid search
for lam in tqdm(param_grid['lam'], desc='lam', position=0):
    for n_splines in tqdm(param_grid['n_splines'], desc='splines', position=1):
        model = PoissonGAM(lam=lam, n_splines=n_splines, max_iter=50, tol=0.05)
        try:
            with suppress_stdout():
                model.fit(X_train, y_train)
            score = model.statistics_['deviance']  # Or another metric
            if score < best_score:
                best_score = score
                best_model = model
        except:
            continue

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")



