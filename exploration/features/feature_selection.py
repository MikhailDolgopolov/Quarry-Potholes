import glob
import json
import random
from collections import Counter
from pprint import pprint

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from exploration.data_read import load_engineered_data
from helpers import split_off_target_cols, train_split_by_column


def select_by_predictors(num_cols:int, ws:int, data_path:str):
    target = "pothole"
    results = []
    iterations = 4
    output_file = f"exploration/features/features_regression_ws{ws}.txt"
    df = load_engineered_data(data_path)
    for _ in trange(iterations//2):
        X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.25)
        model = RandomForestClassifier(n_jobs=4)
        model.fit(X_train, y_train)
        perm_results = permutation_importance(
            model, X_test, y_test, n_repeats=4, scoring='roc_auc', random_state=42
        )
        sorted_idx = perm_results.importances_mean.argsort()[::-1]
        selected_features = list(X_train.columns[sorted_idx][:num_cols])

        # print(f"WS {ws} Sample {i} permutation importance:", selected_features)
        results.extend(selected_features)

    for _ in trange(iterations//2):
        X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.35)
        lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=700)
        lr.fit(X_train, y_train)
        selector = SelectFromModel(lr, prefit=True)
        selected_features_l1 = list(X_train.columns[selector.get_support()][:num_cols])

        # print(f"WS {ws} Sample {i} Log Reg L1:", selected_features_l1)
        results.extend(selected_features_l1)

    count = Counter(results)
    answer = [f'"{k}"' for k, v in count.items() if v >= iterations // 2]
    with open(output_file, "a") as f:
        f.write(','.join(answer))
        f.write('\n')
    return results


def select_by_mannwhitney(num_cols: int, ws: int, data_path: str) -> list[str]:
    """
    Selects a set of 'important' features using the Mann–Whitney U test,
    based on standardized values. For each random sample, the function
    computes the p-value for each feature (comparing the two classes),
    then selects the top num_cols features with the lowest p-values.
    """
    target = "pothole"
    results = []
    output_file = f"exploration/features/features_mannwhitney_ws{ws}.txt"

    # Load data
    df = load_engineered_data(data_path)

    iterations = 3
    # Perform the test multiple times (e.g., 3 random splits).
    for i in trange(iterations, desc="Mann–Whitney iterations"):
        scaler = StandardScaler()
        curr_df = df.sample(frac=0.5, random_state=i)
        X, y = split_off_target_cols(curr_df, target, 1)
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Compute p-values for each feature.
        feature_pvals = {}
        for col in X_scaled.columns:
            # Get values for each group.
            group0 = X_scaled.loc[y == 0, col].dropna()
            group1 = X_scaled.loc[y == 1, col].dropna()
            if len(group0) == 0 or len(group1) == 0:
                feature_pvals[col] = 1.0
            else:
                stat, p = mannwhitneyu(group0, group1, alternative="two-sided")
                feature_pvals[col] = stat
        # Sort features by ascending p-value.
        sorted_features = sorted(feature_pvals, key=lambda x: feature_pvals[x])
        selected_features = sorted_features[:num_cols]
        results.extend(selected_features)

    count = Counter(results)
    answer = [f'"{k}"' for k, v in count.items() if v >= iterations//2]
    with open(output_file, "a") as f:
        f.write(','.join(answer))
        f.write('\n')
    return results

if __name__ == "__main__":
    ws = 7
    # select_by_mannwhitney(num_cols=12, ws=ws, data_path=f'data/engineered/ws30_peaks/rolled{ws}')
    # select_by_predictors(12, ws, f'data/engineered/ws30_peaks/rolled{ws}')
    # pass
    cols = []
    for file_path in glob.glob(f"exploration/features/features_*ws{ws}.txt"):
        with open(file_path, "r") as f:
            for line in f:
                cols.extend(line.strip().split(','))

    count = Counter(cols)
    tries = max(count.values())
    answer = [k for k, v in count.items() if v > tries // 2]
    print(f"{ws=}, {tries=}, {len(answer)=}")
    output_file = f"exploration/features/ws{ws}_features_combined.txt"
    with open(output_file, "w") as f:
        f.write(','.join(answer))
        f.write('\n')
