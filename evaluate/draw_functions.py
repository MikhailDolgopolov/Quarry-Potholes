import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklvq import GLVQ


def LVQ_class_separation(model:GLVQ, prototype_originals: np.ndarray, feature_names):
    # Assign prototypes to classes.
    prototype_labels = model.prototypes_labels_
    prototypes_even = prototype_originals[prototype_labels == 0]
    prototypes_hole = prototype_originals[prototype_labels == 1]

    # Compute mean prototypes.
    mean_prototype_even = np.mean(prototypes_even, axis=0)
    mean_prototype_hole = np.mean(prototypes_hole, axis=0)
    prototype_desc = f'({len(prototypes_even)}, {len(prototypes_hole)})'
    # === Feature Importance Analysis ===
    differences = np.abs(mean_prototype_hole - mean_prototype_even)
    feature_importance = pd.Series(differences, index=feature_names).sort_values(ascending=False)
    top_features = feature_importance.index

    # === Visualization ===
    n_cols = 3  # Number of columns for subplots
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 5 * n_rows))
    fig.suptitle(f"Significant Features and Their Ranges ({prototype_desc} Prototypes)", fontsize=14)
    ax_flat = ax.flatten()

    bar_params = {'alpha': 0.4, 'width': 0.25}
    for i, feature in enumerate(top_features):
        # Plot each prototype for even class.
        for proto in prototypes_even:
            ax_flat[i].bar('even', proto[feature_names == feature], color='blue', **bar_params)
        # Plot each prototype for hole class.
        for proto in prototypes_hole:
            ax_flat[i].bar('hole', proto[feature_names == feature], color='red', **bar_params)

        # Plot mean prototypes.
        ax_flat[i].bar('even', mean_prototype_even[feature_names == feature], color='blue', label='even mean',
                       **bar_params)
        ax_flat[i].bar('hole', mean_prototype_hole[feature_names == feature], color='red', label='hole mean',
                       **bar_params)

        ax_flat[i].set_title(f"{feature}")
        ax_flat[i].set_ylabel("Feature Value")
        ax_flat[i].legend()

    # Hide any unused subplots.
    for j in range(i + 1, len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()