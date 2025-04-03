import json

import numpy as np
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from exploration.data_read import load_preprocessed
from helpers import train_split_by_column
from models.model_registry import predict_with_my_model, predict_with_top_models

target, ws = 'hole', 10
df = load_preprocessed(f'data/{target}{ws}', keep_latlon=False, sample_frac=1)
X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

model_config = {
    'GLVQ': 2,
    'HGBC': 1,
    'SVM': 2
}

def collect_meta_features(X, phase='train'):
    """Collect probabilities from multiple models per type"""
    meta_features = []
    for model_type, n_models in model_config.items():
        try:
            # Get predictions from top n models
            preds = predict_with_top_models(
                X,
                n_models=n_models,
                model_type=model_type,
                return_probas=True,
                strict=False
            )

            # Add each model's probabilities as separate features
            for model_probs in preds:
                meta_features.append(model_probs)

        except Exception as e:
            print(f"Skipping {model_type}: {e}")
            continue
    return np.column_stack(meta_features) if meta_features else None


# Collect meta features
print("Collecting training meta-features...")
X_meta_train = collect_meta_features(X_train)
print("Collecting test meta-features...")
X_meta_test = collect_meta_features(X_test)

if X_meta_train is None or X_meta_test is None:
    raise ValueError("Could not collect any base model predictions")

# Train meta-classifier with cross-validation
print("\nTraining meta-classifier...")
meta_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Cross-validate on training data
# cv_preds = cross_val_predict(meta_clf, X_meta_train, y_train, cv=5, method='predict_proba')[:, 1]
# cv_preds_binary = (cv_preds >= 0.5).astype(int)

# Final training
meta_clf.fit(X_meta_train, y_train)

# Evaluate
test_meta_probs = meta_clf.predict_proba(X_meta_test)[:, 1]
test_meta_pred = (test_meta_probs >= 0.5).astype(int)

# Performance comparison
print("\nMeta-Predictor Performance Analysis:")
print("=" * 50)

# Evaluate base models
model_scores = {}
for model_type in model_config:
    try:
        pred = predict_with_my_model(X_test, model_type=model_type)
        model_scores[model_type] = f1_score(y_test, pred)
    except Exception as e:
        print(f"Skipping {model_type} evaluation: {e}")
        continue

model_scores['Meta-Test'] = f1_score(y_test, test_meta_pred)

# Print comparison
print("{:<12} {:<10}".format('Model', 'F1 Score'))
print("-" * 22)
for model, score in sorted(model_scores.items(), key=lambda x: -x[1]):
    print("{:<12} {:.4f}".format(model, score))

# Detailed metrics
print("\nMeta-Predictor Test Report:")
print(classification_report(y_test, test_meta_pred))

# Meta-Predictor Test Report:
#               precision    recall  f1-score   support
#
#          0.0       0.77      0.86      0.81     19940
#          1.0       0.67      0.54      0.60     11146
#
#     accuracy                           0.74     31086
#    macro avg       0.72      0.70      0.70     31086
# weighted avg       0.73      0.74      0.73     31086