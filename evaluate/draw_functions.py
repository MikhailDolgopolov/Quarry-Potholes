from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from sklvq import GLVQ


def lvq_class_separation(model: GLVQ, X, feature_names, accuracy=None, save_path: str | Path = None):
    """
    Visualizes the prototype ranges (as semi-transparent background boxes with dashed borders)
    and predicted data IQR (as foreground boxes with solid borders) for each feature and class.
    """
    feature_names = np.array(feature_names)
    unique_classes = np.unique(model.prototypes_labels_)
    class_counts = {cls: np.sum(model.prototypes_labels_ == cls) for cls in unique_classes}
    counts_str = ", ".join([f"{cls}: {class_counts[cls]}" for cls in sorted(unique_classes)])
    prototype_desc = f"({counts_str})"

    y_pred = model.predict(X)

    # Compute feature importance (range between class means)
    mean_prototypes = {
        cls: np.mean(model.prototypes_[model.prototypes_labels_ == cls], axis=0)
        for cls in unique_classes
    }
    feature_importance = pd.Series(
        [np.ptp([mean_prototypes[cls][j] for cls in unique_classes]) for j in range(len(feature_names))],
        index=feature_names
    ).sort_values(ascending=False)
    top_features = feature_importance.index

    # Colors for each class
    n_classes = len(unique_classes)
    cmap = plt.get_cmap('Set1')
    colors = {cls: cmap(i / n_classes) for i, cls in enumerate(sorted(unique_classes))}

    # Setup subplots
    n_cols = 3
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(f"Feature Separation Analysis {prototype_desc}", fontsize=16, fontweight="bold")
    if accuracy is not None:
        fig.text(0.5, 0.93, f"Accuracy: {accuracy:.2f}", fontsize=12, ha="center", style="italic")
    axes = axes.flatten()

    # Plot for each feature
    for i, feature in enumerate(top_features):
        ax = axes[i]
        feat_idx = np.where(feature_names == feature)[0][0]

        for j, cls in enumerate(sorted(unique_classes)):
            x_center = j
            proto_width = 0.5
            pred_width = 0.3

            prototypes = model.prototypes_[model.prototypes_labels_ == cls][:, feat_idx]
            predicted_points = X.loc[y_pred == cls, feature].values if isinstance(X, pd.DataFrame) else X[
                                                                                                            y_pred == cls][
                                                                                                        :, feat_idx]
            # Plot prototypes (range)
            if len(prototypes) > 0:
                proto_min, proto_max = np.min(prototypes), np.max(prototypes)
                ax.add_patch(Rectangle(
                    (x_center - proto_width / 2, proto_min),
                    proto_width,
                    proto_max - proto_min,
                    facecolor=colors[cls],
                    alpha=0.4,
                    edgecolor=colors[cls],
                    linestyle='dashed',
                    linewidth=1
                ))
                if len(prototypes) == 1:
                    ax.hlines(prototypes[0], x_center - proto_width / 2, x_center + proto_width / 2,
                              colors=colors[cls], linewidth=3, alpha=0.7)

            # Plot predicted data (IQR)
            if len(predicted_points) > 0:
                q25, med, q75 = np.percentile(predicted_points, [25, 50, 75])
                ax.add_patch(Rectangle(
                    (x_center - pred_width / 2, q25),
                    pred_width,
                    q75 - q25,
                    facecolor=colors[cls],
                    alpha=0.3,
                    edgecolor='black',
                    linewidth=1
                ))
                ax.hlines(med, x_center - pred_width / 2, x_center + pred_width / 2, colors='black', linewidth=2)

        ax.set_xticks(range(len(unique_classes)))
        ax.set_xticklabels([str(cls) for cls in sorted(unique_classes)])
        ax.set_title(feature, fontsize=12)
        ax.grid(True, axis='y', linestyle=':', alpha=0.7)

    # Create a legend
    class_legend = [Patch(facecolor=colors[cls], label=f'Class {cls}') for cls in sorted(unique_classes)]
    proto_legend = Patch(facecolor='gray', alpha=0.4, linestyle='dashed', label="Prototypes (Range)")
    pred_legend = Patch(facecolor='gray', alpha=0.3, edgecolor='black', label="Predicted Data (IQR)")
    fig.legend(handles=class_legend + [proto_legend, pred_legend],
               loc='upper right', bbox_to_anchor=(0.98, 0.88), fontsize=10)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close(fig)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_feature_ranges(df: pd.DataFrame, target: str = 'pothole', f=None):
    """
    Create a grid of box plots for the given features.
    Each subplot displays a box plot of the feature distribution
    grouped by target class, with different colors per class.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the feature columns and the target column.
    target : str, optional
        The name of the target column. Defaults to 'pothole'.
    """
    if f is None:
        f = df.columns

    df = df.copy()

    for feat in f:
        if df[feat].dtype == object:
            df[feat] = pd.to_numeric(df[feat], errors='ignore')

    df[target] = pd.Categorical(df[target])
    class_names = df[target].cat.categories
    palette = sns.color_palette("husl", len(class_names))  # bright, distinct colors

    n_features = len(f)
    n_cols = int(np.sqrt(n_features))
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, feat in enumerate(f):
        ax = axes[i]
        sns.boxplot(x=target, y=feat, data=df, ax=ax, showfliers=False, palette=palette, hue=target)
        ax.set_title(feat, fontsize=12)
        ax.set_ylabel("Value")
        ax.set_xlabel("")
        ax.grid(True, axis='y', linestyle=':', alpha=0.7)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

