import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from tqdm import trange

from exploration.data_read import load_prepared
from helpers import train_split_by_column

if __name__ == '__main__':
    target = 'hole'
    results = {}

    for ws in [5, 7, 10]:
        ws_key = f"WS_{ws}"
        results[ws_key] = {
            "permutation_importance": [],
            "log_reg_l1": []
        }

        for i in range(3):
            df = load_prepared(f'data/{target}{ws}', keep_latlon=False, sample_frac=0.3)
            X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.5)
            model = RandomForestClassifier(n_jobs=4)
            model.fit(X_train, y_train)
            perm_results = permutation_importance(
                model, X_test, y_test, n_repeats=5, scoring='roc_auc', random_state=42
            )
            sorted_idx = perm_results.importances_mean.argsort()[::-1]
            selected_features = list(X_train.columns[sorted_idx][:10])

            # print(f"WS {ws} Sample {i} permutation importance:", selected_features)
            results[ws_key]["permutation_importance"].append(selected_features)

        for i in range(3):
            df = load_prepared(f'data/{target}{ws}', keep_latlon=False, sample_frac=0.5)
            X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.5)
            lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=700)
            lr.fit(X_train, y_train)
            selector = SelectFromModel(lr, prefit=True)
            selected_features_l1 = list(X_train.columns[selector.get_support()][:10])

            # print(f"WS {ws} Sample {i} Log Reg L1:", selected_features_l1)
            results[ws_key]["log_reg_l1"].append(selected_features_l1)

        # print("\n")

    with open("exploration/feature_selection_sets.json", "w") as f:
        json.dump(results, f, indent=4)