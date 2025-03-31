from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklvq import GLVQ


def LVQ_class_separation(model: GLVQ, prototype_originals: np.ndarray, feature_names, accuracy=None,
                         save_path: str | Path = None):
    feature_names = np.array(feature_names)
    # Get unique classes and their counts
    unique_classes = np.unique(model.prototypes_labels_)
    class_counts = {cls: np.sum(model.prototypes_labels_ == cls) for cls in unique_classes}
    counts_str = ", ".join([f"{cls}: {class_counts[cls]}" for cls in sorted(unique_classes)])
    prototype_desc = f"({counts_str})"

    # Compute mean prototypes for each class.
    mean_prototypes = {cls: np.mean(prototype_originals[model.prototypes_labels_ == cls], axis=0)
                       for cls in unique_classes}
    # Compute feature importance as the peak-to-peak range among class means.
    feature_importance = pd.Series(
        [np.ptp([mean_prototypes[cls][j] for cls in unique_classes]) for j in range(len(feature_names))],
        index=feature_names
    ).sort_values(ascending=False)
    top_features = feature_importance.index

    # Create a color mapping for classes using a colormap.
    n_classes = len(unique_classes)
    cmap = plt.get_cmap('Set1')
    colors = {cls: cmap(i / n_classes) for i, cls in enumerate(sorted(unique_classes))}

    # === Visualization Enhancements ===
    n_cols = 3  # Number of columns for subplots
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(f"Significant Features and Their Ranges {prototype_desc}", fontsize=16, fontweight="bold")

    # Subtitle with accuracy (if available)
    if accuracy is not None:
        fig.text(0.5, 0.93, f"Overall Accuracy: {accuracy:.2f}", fontsize=12, ha="center", style="italic")

    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        ax = axes[i]
        # Get index of the feature in the array.
        feat_idx = np.where(feature_names == feature)[0][0]

        # Collect values for each class for the current feature.
        data = []
        for cls in sorted(unique_classes):
            values = prototype_originals[model.prototypes_labels_ == cls][:, feat_idx]
            data.append(values)

        # Positions for the boxplots (one per class)
        positions = list(range(len(unique_classes)))
        bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=False)

        # Set the color for each box according to its class.
        for j, box in enumerate(bp['boxes']):
            cls = sorted(unique_classes)[j]
            box.set_facecolor(colors[cls])

        ax.set_xticks(positions)
        ax.set_xticklabels([str(cls) for cls in sorted(unique_classes)])
        ax.set_title(f"{feature}")
        ax.grid(True, axis='y', linestyle="--", alpha=0.6)

    # Create a custom legend for the classes
    legend_patches = [Patch(color=colors[cls], label=f"Class {cls}") for cls in sorted(unique_classes)]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=10)

    # Hide any unused subplots.
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=0.4, w_pad=0.2)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def visualize_classes(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Count the frequency of each class
    class_counts: pd.Series = df['class'].value_counts().sort_index()

    # Create bar plot
    class_counts.plot(kind='bar', edgecolor='black', color='blue')

    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Count of Each Class in the Dataset')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with their counts
    for i, count in enumerate(class_counts):
        plt.text(i, count + 0.02 * max(class_counts), str(count), ha='center', va='bottom')

    plt.show()
