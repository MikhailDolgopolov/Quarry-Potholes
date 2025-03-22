import glob
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import compute_sample_weight

from exploration.data_read import load_prepared


# filename='models/HGBR_[l2_regularization0.5][losssquared_error][max_depth6][min_samples_leaf100]_23.pkl'
# out_fr = 0.1
# main_fr = 0.5


def evaluate_classification(X, y_test, filename) -> Tuple[pd.DataFrame, LinearRegression, float]:
    # Load the model
    with open(filename, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X)

    # Create a dataframe with true and predicted values
    results_df = pd.DataFrame({
        'true_class': y_test,
        'prediction': y_pred
    })

    # Compute RMSE with balanced sample weights
    weights = compute_sample_weight('balanced', y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=weights))

    # Fit linear regression on all true vs. predicted points
    reg = LinearRegression()
    reg.fit(y_test.values.reshape(-1, 1), y_pred, sample_weight=weights)

    return results_df, reg, rmse


def draw_linear(results_df, linreg, rmse, name='Linearity'):
    unique_classes = np.sort(results_df['true_class'].unique())
    #
    # for value in unique_classes:
    #     fig, ax = plt.subplots(figsize=(12, 7))
    #     dist = results_df[results_df['true_class']==value]
    #     sns.histplot(dist['prediction'])
    #     plt.title(f'Distribution on class {value}')
    #     # plt.show()
    #     plt.savefig(f'images/distrs/model{round(rmse)}_class{value}.png')
    #     plt.close(fig)
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))


    # Prepare data for boxplots: list of predictions for each true_class
    boxplot_data = [results_df[results_df['true_class'] == cls]['prediction'].values
                    for cls in unique_classes]

    # Plot boxplots at the actual true_class positions
    ax.violinplot(boxplot_data, positions=unique_classes, widths=10, showmeans=True, showmedians=True)

    # Add regression line over the actual true_class range
    x_min, x_max = unique_classes.min(), unique_classes.max()
    x_range = np.linspace(x_min, x_max, 100)  # Smooth line across the range
    ax.plot(x_range, linreg.predict(x_range.reshape(-1, 1)), 'r--', lw=2,
            # label=f'y = {linreg.coef_[0]:.2f}x + {linreg.intercept_:.2f}'
            )

    # Formatting
    ax.set_xlabel('True Class', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'{name} Evaluation\nRMSE: {rmse:.4f}', fontsize=14)
    plt.xlim([x_min-20, x_max+20])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    target = 'class'
    df = load_prepared(f'data/{target}10', keep_latlon=True, sample_frac=1)
    col='rel'
    df=df.drop(columns=[col, 'lat', 'lon'], errors='ignore')
    X, y_test = df.drop(columns=[target]), df[target]

    models_path = 'models/*.pkl'
    model_files = glob.glob(models_path)

    for f in model_files:
        stats, reg, score = evaluate_classification(X, y_test, f)
        draw_linear(stats, reg, score, name=f)


