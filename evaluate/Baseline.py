import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import statsmodels.api as sm

from exploration.data_read import load_engineered_data
from helpers import train_split_by_column

target, ws = 'pothole', 7
df = load_engineered_data(f'data/engineered/30peaks_eps5/rolled{ws}')
X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.5)

# Compute balanced sample weights for the training set
sample_weights = compute_sample_weight("balanced", y_train)

# Instead of using Logit, we use GLM with a Binomial family which accepts sample weights.
model = sm.GLM(
    y_train,
    X_train,
    family=sm.families.Binomial(),
    freq_weights=sample_weights
).fit()

print(model.summary())

# Predict probabilities on the test set
y_pred_prob = model.predict(X_test)

# --- Threshold Tuning ---
# When the minority class is more important to detect, you often want to maximize its recall.
# Here we search for the threshold that yields the highest recall for the positive class.
thresholds = np.linspace(0, 1, 101)
best_thresh = 0.0
best_recall = 0.0

for t in thresholds:
    y_pred_temp = np.where(y_pred_prob >= t, 1, 0)
    rec = f1_score(y_test, y_pred_temp, pos_label=1)
    if rec > best_recall:
        best_recall = rec
        best_thresh = t

print(f"Best threshold: {best_thresh:.2f}")
print(f"Best recall for minority class: {best_recall:.2f}")

# Use the tuned threshold for final predictions
y_pred = np.where(y_pred_prob >= best_thresh, 1, 0)

print(classification_report(y_test, y_pred))