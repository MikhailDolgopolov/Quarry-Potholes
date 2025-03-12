import os
import pickle
from typing import Any

import numpy as np
import pygam
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, floating, complexfloating
from numpy._typing import _64Bit
from pygam import LogisticGAM
from pygam.terms import SplineTerm
from scipy.special import expit  # Logistic function
from sklearn.model_selection import train_test_split
from data_read import load_prepared

big_df = load_prepared('data/prepared10')
train_df, test_df = train_test_split(big_df, test_size=0.2)
X_train, y_train = train_df.drop(columns=['hole']), train_df['hole']
X_test, y_test = test_df.drop(columns=['hole']), test_df['hole']

def get_model(name:str, train_new=False) -> LogisticGAM:
    filename = f'models/{name}.pkl'

    if train_new or not os.path.exists(filename):
        model = LogisticGAM(
            n_splines=15,
            lam=0.6,
            max_iter=50,
            tol=0.01,
        )
        print('Fitting a LogisticGAM...')
        model.fit(X_train, y_train)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    else:
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filename}: {e}")

    return model


def draw_partial_deps(m: LogisticGAM):
    # Get feature names (excluding the target 'hole')
    feature_names = big_df.drop(columns=['hole']).columns
    n_features = len(feature_names)

    # Create a roughly square grid of subplots
    n_cols = int(np.ceil(np.sqrt(n_features)))
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5), sharey=True)
    axes = axes.flatten()  # Flatten for easy indexing

    for i, feature in enumerate(feature_names):
        # Skip non-spline terms (e.g., intercept)
        if i >= len(m.terms) or not isinstance(m.terms[i], SplineTerm):
            continue

        # # Get the observed range of the feature
        feature_data = big_df[feature].values
        min_val = feature_data.min()
        max_val = feature_data.max()

        # Generate a grid of 100 points within the observed range
        XX = m.generate_X_grid(term=i, n=100)
        pdep, confi = m.partial_dependence(term=i, X=XX, width=0.95)

        # Convert to probability
        pdep_prob = expit(pdep)
        confi_lower_prob = expit(confi[:, 0])
        confi_upper_prob = expit(confi[:, 1])

        # Plot
        ax = axes[i]
        ax.plot(XX[:, m.terms[i].feature], pdep_prob, label='Partial Dependence')
        ax.fill_between(XX[:, m.terms[i].feature], confi_lower_prob, confi_upper_prob,
                        alpha=0.2, label='95% CI')
        ax.set_title(f'Partial Dependence of {feature}')
        ax.set_ylabel('Probability')
        ax.set_ylim([0, 1])
        ax.set_xlim(min_val, max_val)  # Restrict to observed range
        ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=5)
    plt.show()

draw_partial_deps(get_model('gam10'))