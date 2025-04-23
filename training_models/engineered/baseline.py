import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import statsmodels.api as sm

from exploration.data_read import load_engineered_data
from exploration.features.Separability import anomaly_features
from helpers import train_split_by_column


target, ws = 'pothole', 7
df = load_engineered_data(f'data/engineered/combined/rolled{ws}')
X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.2)
X_train, X_test = X_train[anomaly_features], X_test[anomaly_features]
print(len(X_train.columns))
# Compute balanced sample weights for the training set
sample_weights = compute_sample_weight("balanced", y_train)

# Fit a GLM with a Binomial family which accepts sample weights.
# Note: exog must be numeric; ensure X_train is numeric.
# model = LogisticRegression()

# Fit the model to the training data
# model.fit(X_train, y_train)
model = sm.GLM(
    endog=y_train,
    exog=X_train,
    family=sm.families.Binomial(),
    freq_weights=sample_weights
).fit()

# Predict probabilities on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# --- Threshold Tuning ---
# We tune the threshold to maximize macro-average F1
# thresholds = np.linspace(0, 1, 101)
# best_thresh = 0.0
# best_macro_f1 = 0.0
#
# for t in thresholds:
#     y_pred_temp = np.where(y_pred_prob >= t, 1, 0)
#     macro_f1 = f1_score(y_test, y_pred_temp, average='macro')
#     if macro_f1 > best_macro_f1:
#         best_macro_f1 = macro_f1
#         best_thresh = t
#
# print(f"Best threshold: {best_thresh:.2f}")
# print(f"Best Macro F1: {best_macro_f1:.2f}")

# Use the tuned threshold for final predictions


# Best threshold: 0.61
# Best Macro F1: 0.57
#               precision    recall  f1-score   support
#
#            0       0.89      0.88      0.88     27324
#            1       0.23      0.26      0.25      3951
#
#     accuracy                           0.80     31275
#    macro avg       0.56      0.57      0.57     31275
# weighted avg       0.81      0.80      0.80     31275