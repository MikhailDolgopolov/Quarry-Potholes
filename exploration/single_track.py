import matplotlib.pyplot as plt
import pandas as pd
from Tools.scripts.generate_re_casefix import alpha

from exploration.data_read import load_prepared, read_new_points
from geospacial.load_latlon import filter_reliable_potholes
from helpers import train_split_by_column
from training_models.GradBoosting import train_evaluate


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

if __name__ == '__main__':
    # target, ws = 'class', 10
    # df_full = load_prepared(f'data/{target}{ws}', keep_latlon=True, sample_frac=1)
    # print(len(df_full))

    df = read_new_points('data/routes/route5/2_w.csv')
    # print(df.describe())
    #
    plt.figure(figsize=(10, 6))
    plt.plot([0] * len(df),  '--', color='gray',)
    plt.plot(df['acc_X'], label='X', alpha=0.8)
    plt.plot(df['acc_Y'], label='Y', alpha=0.8)
    plt.plot(df['acc_Z']*10, label='10х Acceleration Z', alpha=0.8)


    # Add labels and title
    plt.xlabel('1Hz readings')
    plt.ylabel('Acceleration (m/s2)')
    plt.title('Accelerometers')
    plt.legend()
    plt.show()



