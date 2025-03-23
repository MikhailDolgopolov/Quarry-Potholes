import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from evaluate.other import manage_models, evaluate_classification
from exploration.data_read import load_prepared
from helpers import discretize_to_levels


def draw_linear(results_df, linreg, rmse, name='Linearity'):
    # Setup figure and colormap
    unique_classes = np.sort(results_df['true_class'].unique())
    cmap = LinearSegmentedColormap.from_list('severity', ['green', 'yellow', 'red'])
    norm = plt.Normalize(vmin=unique_classes.min(), vmax=unique_classes.max())

    fig, ax = plt.subplots(figsize=(12, 7))
    # results_df['prediction'] = discretize_to_levels(results_df['prediction'].to_numpy(), np.linspace(0, 150, 15))
    # Prepare and plot violin distributions
    boxplot_data = [results_df[results_df['true_class'] == cls]['prediction'].values
                    for cls in unique_classes]
    violins = ax.violinplot(
        boxplot_data,
        positions=unique_classes,
        widths=np.diff(unique_classes).min(),
        showmeans=True,
        showmedians=True
    )

    # Color violins by true class severity
    for pc, cls in zip(violins['bodies'], unique_classes):
        pc.set_facecolor(cmap(norm(cls)))
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    # Style mean/median lines
    violins['cmeans'].set_color('blue')
    violins['cmedians'].set_color('black')

    # Add regression line
    x_min, x_max = unique_classes.min(), unique_classes.max()
    x_range = np.linspace(x_min, x_max, 100)
    ax.plot(x_range, linreg.predict(x_range.reshape(-1, 1)),
            '--', lw=2, c='darkblue',
            label=f'y = {linreg.coef_[0]:.2f}x + {linreg.intercept_:.2f}')

    # # Add colorbar legend
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax)
    # cbar.set_label('True Severity Level')

    # Formatting
    ax.set_xlabel('True Pothole Severity Class', fontsize=12)
    ax.set_ylabel('Predicted Severity Value', fontsize=12)
    ax.set_title(f'{name} Evaluation\nRMSE: {rmse:.4f}', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_min - 20, x_max + 20])

    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    target = 'class'
    df = load_prepared(f'data/class10', keep_latlon=True, sample_frac=1)
    df=df.drop(columns=['lat', 'lon'], errors='ignore')
    X, y_test = df.drop(columns=[target]), df[target]
    models_path = 'models/HGBR*.pkl'
    # manage_models(X, y_test, models_path)
    #
    model_file = glob.glob(models_path)[1]

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    pred = np.clip(model.predict(X), y_test.min(), y_test.max())

    df[target] = pred
    output_path = 'data/predicted.csv'
    df.to_csv(output_path, index=False, sep=';')
    # stats, reg, score = evaluate_classification(X, y_test, model_file)
    # draw_linear(stats, reg, score, name=model_file)


