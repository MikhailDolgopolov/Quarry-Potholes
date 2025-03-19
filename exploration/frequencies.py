import matplotlib.pyplot as plt
from exploration.data_read import load_prepared
from geospacial.load_latlon import filter_reliable_potholes
from helpers import train_split_by_column
from training_models.GradBoosting import train_evaluate


def visualize_classes(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Count the frequency of each class
    class_counts = df['class'].value_counts().sort_index()

    # Create bar plot
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')

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

if __name__ == '__main__':
    target, ws = 'class', 0
    df_full = load_prepared(f'data/raw{ws}', keep_latlon=True, sample_frac=0.2)

    col='rel'
    params = {'cluster_samples': 5,
             'eps': 0.02,
             'hole_threshold': 25,
             'positive_class_ratio': 0.5,
             'reports': 10}
    # df_reliable = filter_reliable_potholes(df_full, **params, reliable_col=col)



