import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight

from evaluate.draw_functions import plot_feature_ranges
from exploration.data_read import load_plain_data, straight_predictors

# Load data from input folder.
# input_folder = Path('data/relabeled/extremes_w10')
input_folder = Path('data/relabeled/extremes_w10')
data = load_plain_data(input_folder)

print(straight_predictors)
# X, y = data[straight_predictors], data['pothole']
X, y = data[straight_predictors], np.where(data['severity'] >=0.3, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=5, stratify=y
)

# 2) Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 3) Compute balanced sample weights on TRAIN only
train_weights = compute_sample_weight('balanced', y_train)

# 4) Fit logistic regression with sample weights
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train, sample_weight=train_weights)

# 5) Get predicted probabilities on TEST
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# 6) Tune threshold for best macro‑F1
thresholds = np.linspace(0, 1, 101)
best_thresh, best_macro_f1 = 0.5, 0.0

for t in thresholds:
    y_temp = (y_pred_prob >= t).astype(int)
    m = f1_score(y_test, y_temp, average='macro')
    if m > best_macro_f1:
        best_macro_f1, best_thresh = m, t

print(f"Best threshold = {best_thresh:.2f}, macro‑F1 = {best_macro_f1:.3f}")

with open(f'models/LogReg/Plain_{best_thresh*100:.0f}.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

# 7) Final predictions at tuned threshold
y_final = (y_pred_prob >= best_thresh).astype(int)
print(classification_report(y_test, y_final))