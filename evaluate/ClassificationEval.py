import glob
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from exploration.data_read import load_prepared


# filename='models/HGBR_[l2_regularization0.5][losssquared_error][max_depth6][min_samples_leaf100]_23.pkl'
# out_fr = 0.1
# main_fr = 0.5


def evaluate_classification(X, y_test, filename, main_fr=0.5, out_fr=0.1)->Tuple[float, pd.DataFrame, LinearRegression]:
    with open(filename, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    results_df = pd.DataFrame({
        'true_class': y_test,
        'prediction': y_pred
    })
    quantile_stats = (
        results_df.groupby('true_class', observed=False)['prediction']
        .agg(
            edge1=lambda x: x.quantile(out_fr),
            strip1=lambda x: x.quantile(0.5 - main_fr * 0.5),
            mean=lambda x: x.mean(),
            strip2=lambda x: x.quantile(0.5 + main_fr * 0.5),
            edge2=lambda x: x.quantile(1 - out_fr),
            count=lambda x: x.count()
        )
        .reset_index()
        .sort_values('true_class')
    )
    short, long = 'short', 'long'
    for i in [1, 2]:
        quantile_stats[f'{long}{i}'] = quantile_stats[f'edge{i}'] - quantile_stats[f'strip{i}']
        quantile_stats[f'{short}{i}'] = quantile_stats[f'strip{i}'] - quantile_stats[f'mean']
    cols = [''.join(t) for t in list(product([short, long], map(str, [1, 2])))]
    point_cols = []
    for i in range(4):
        quantile_stats[f'point{i + 1}'] = quantile_stats['mean'] + quantile_stats[cols[i]]
        point_cols.append(f'point{i + 1}')
    # print(quantile_stats)
    cols.extend([''.join(t) for t in list(product(['strip', 'edge'], map(str, [1, 2])))])
    wide = quantile_stats.drop(columns=cols)
    points = pd.melt(
        wide,
        id_vars=['true_class', 'count'],
        value_vars=point_cols,
        var_name='quantile',
        value_name='y'
    ).sort_values(['true_class'])
    classes = np.array(points['true_class']).reshape(-1, 1)
    predicted_edges = points['y']
    weights = points['count']  # Get weights from melted data
    # Fit weighted regression
    reg = LinearRegression()
    # reg.fit(classes, predicted_edges, sample_weight=weights)
    reg.fit(classes, predicted_edges)
    linear_result = reg.predict(classes)
    RMSE = np.sqrt(mean_squared_error(linear_result, predicted_edges, sample_weight=weights))
    return RMSE, quantile_stats, reg


def draw_linear(quantile_df,linreg:LinearRegression, rmse, name='Linearity'):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Custom boxplot parameters
    boxprops = dict(facecolor='skyblue', linewidth=2)
    whiskerprops = dict(color='navy', linestyle='--')
    medianprops = dict(color='gold', linewidth=2)

    # quantile_df = quantile_df.set_index('true_class')
    # Create custom boxplots using your quantiles
    for idx, row in quantile_df.iterrows():
        # Box from strip1 to strip2
        ax.fill_between(
            [row['true_class'] - 10,
             row['true_class'] + 10],
            row['strip1'],
            row['strip2'],
            **boxprops
        )

        # Median line
        ax.hlines(
            row['mean'],
            row['true_class'] - 10,
            row['true_class'] + 10,
            **medianprops
        )

        # Whiskers
        ax.vlines(
            row['true_class'],
            row['edge1'],
            row['edge2'],
            **whiskerprops
        )

    # Add regression line
    x_range = np.linspace(quantile_df['true_class'].min(),
                          quantile_df['true_class'].max(), 100)
    ax.plot(x_range, linreg.predict(x_range.reshape(-1, 1)),
            'r--', lw=2, label=f'y = {linreg.coef_[0]:.2f}x + {linreg.intercept_:.2f}')

    # Formatting
    ax.set_xlabel('True Class', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'{name} Evaluation\nRMSE: {rmse}', fontsize=14)
    ax.legend()
    plt.xticks(quantile_df['true_class'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def manage_models(models_path='models/*.pkl'):
    model_files = glob.glob(models_path)
    thresholds = np.arange(0.3, 0.7, 0.04)
    results = {}

    for model_path in tqdm(model_files):
        # print(type(model_path))

        model_name = os.path.basename(model_path).split('\\')[0]
        results[model_name] = []

        for i in thresholds:
            rmse, _, _ = evaluate_classification(X, y_test, model_path, i)
            results[model_name].append(rmse)

    mins = {n: np.min(score) for n, score in results.items()}
    # Rank models by their minimum RMSE (best to worst)
    sorted_mins = sorted(mins.items(), key=lambda x: x[1])

    # Print model rankings
    print("\nModel Rankings (Best to Worst):")
    print("Rank | Model Name".ljust(40) + " | Min RMSE")
    print("-" * 55)
    for rank, (model_name, min_score) in enumerate(sorted_mins, 1):
        print(f"{rank:4} | {model_name[:35]:35} | {min_score:.4f}")

    # Get user input for deletion
    try:
        num_to_delete = int(input("\nEnter number of worst models to delete (0 to cancel): "))
        if num_to_delete <= 0:
            return
    except ValueError:
        print("Invalid input. No models deleted.")
        return

    # Get worst performers to delete
    to_delete = sorted_mins[-num_to_delete:]

    # Confirm deletion
    print("\nWARNING: These models will be permanently deleted:")
    for model_name, score in to_delete:
        print(f"- {model_name} (RMSE: {score:.4f})")

    confirm = input("\nConfirm deletion? (y/n): ").lower()
    if confirm != 'y':
        print("Deletion canceled.")
        return

    # Delete files
    deleted_count = 0
    for model_name, _ in to_delete:
        # Find matching model file
        for model_path in model_files:
            if os.path.basename(model_path) == model_name:
                try:
                    os.remove(model_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {model_name}: {str(e)}")
                break

    print(f"\nSuccessfully deleted {deleted_count}/{num_to_delete} models")

if __name__ == '__main__':
    target = 'class'
    df = load_prepared(f'data/{target}10', keep_latlon=True, sample_frac=1)
    col='rel'
    params = {'cluster_samples': 20,
             'eps': 0.01,
             'hole_threshold': 30,
             'positive_class_ratio': 0.3,
             'reports': 30}
    # df = filter_reliable_potholes(df, **params, reliable_col=col)
    # df = df[df[col]]
    df=df.drop(columns=[col, 'lat', 'lon'], errors='ignore')
    X, y_test = df.drop(columns=[target]), df[target]

    # manage_models()

    models_path = 'models/*.pkl'
    model_files = glob.glob(models_path)[:1]

    for f in model_files:
        score, stats, reg = evaluate_classification(X, y_test, filename=f)
        draw_linear(stats, reg, score, name=f)


