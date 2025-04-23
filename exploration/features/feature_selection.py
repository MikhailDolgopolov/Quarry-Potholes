import glob
import json
import os
import random
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from exploration.data_read import load_engineered_data, load_plain_data
from helpers import split_off_target_cols, train_split_by_column


def select_by_predictors(num_cols:int, ws:int, data_path:str):
    target = "pothole"
    results = []
    iterations = 4
    output_file = f"exploration/features/features_regression_ws{ws}.txt"
    if not os.path.exists(output_file):
        open(output_file, 'w').close()
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


def select_by_mannwhitney(num_cols: int, file_name: str, df: pd.DataFrame) -> list[str]:
    """
    Selects a set of 'important' features using the Mann–Whitney U test,
    based on standardized values. For each random sample, the function
    computes the p-value for each feature (comparing the two classes),
    then selects the top num_cols features with the lowest p-values.
    """
    if 'pothole' not in df.columns:
        df['pothole'] = np.where(df['severity'] >= 0.3, 1, 0)
        df.drop('severity', axis=1, inplace=True)
    target = "pothole"
    results = []
    output_file = f"exploration/features/{file_name}.txt"
    if not os.path.exists(output_file):
        open(output_file, 'w').close()
    # Load data


    iterations = 3
    # Perform the test multiple times (e.g., 3 random splits).
    for i in trange(iterations, desc="Mann–Whitney iterations"):
        scaler = StandardScaler()
        curr_df = df.sample(frac=0.5, random_state=i)
        X, y = split_off_target_cols(curr_df, target, 1)
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Compute p-values for each feature.
        feature_stats = {}
        for col in X_scaled.columns:
            # Get values for each group.
            group0 = X_scaled.loc[y == 0, col].dropna()
            group1 = X_scaled.loc[y == 1, col].dropna()
            if len(group0) == 0 or len(group1) == 0:
                feature_stats[col] = 1.0
            else:
                stat, p = mannwhitneyu(group0, group1, alternative="two-sided")
                feature_stats[col] = stat
        # Sort features by ascending stat.
        sorted_features = sorted(feature_stats, key=lambda x: feature_stats[x])
        selected_features = sorted_features[:num_cols]
        results.extend(selected_features)

    count = Counter(results)
    answer = [f'"{k}"' for k, v in count.items() if v >= iterations//2]
    with open(output_file, "a") as f:
        f.write(','.join(answer))
        f.write('\n')
    return results

if __name__ == "__main__":
    ano = load_plain_data('data/isoforest/2.0_rolled10')
    ori = load_plain_data('data/rolled/extremes_w10_norm_rolled10')
    ano.drop(['lat', 'lon'], axis=1, inplace=True)
    ori.drop(['lat', 'lon'], axis=1, inplace=True)
    # print(ano.columns)
    # print(ori.columns)
    select_by_mannwhitney(10, 'features_U_pressure', ori)
    select_by_mannwhitney(10, 'features_U_anomalies', ano)
    # ews =30
    # rws = 10
    # df = load_plain_data(f'data/rolled/extremes_w{ews}_norm_rolled{rws}')
    # df.drop(['lat', 'lon'], axis=1, inplace=True)
    # X, y = split_off_target_cols(df, 'severity', 1)
    # from sklearn.feature_selection import SelectKBest, f_regression
    #
    # selector = SelectKBest(score_func=f_regression, k=15)
    # X_selected = selector.fit_transform(X, y)
    # feature_scores = pd.DataFrame({
    #     'Feature': X.columns,
    #     'Score': selector.scores_,
    #     'P-value': selector.pvalues_
    # }).sort_values('Score', ascending=False).reset_index(drop=True)
    # feature_scores = feature_scores[feature_scores['P-value']<1e-5]
    # # print(feature_scores)
    #
    # significant_features = feature_scores['Feature'].tolist()  # From your existing code
    #
    # # Subset your original DataFrame to keep only significant features
    # X_filtered = X[significant_features].copy()  # Use .copy() to avoid SettingWithCopyWarning
    #
    # from statsmodels.stats.outliers_influence import variance_inflation_factor
    #
    # # Calculate VIF for each feature
    # vif_data = pd.DataFrame()
    # vif_data["Feature"] = X_filtered.columns
    # vif_data["VIF"] = [
    #     variance_inflation_factor(X_filtered.values, i)
    #     for i in range(X_filtered.shape[1])
    # ]
    #
    #
    # vif_data = vif_data[vif_data['VIF']<5].sort_values("VIF", ascending=True).reset_index(drop=True)
    # print(vif_data)
    # answer = [f'"{k}"' for k in vif_data['Feature'].values]
    # output_file = f"exploration/features/features_ex{ews}_vif.txt"
    # if not os.path.exists(output_file):
    #     with open(output_file, 'w') as f:
    #         pass
    # with open(output_file, "a") as f:
    #     f.write(','.join(answer))
    #     f.write('\n')

