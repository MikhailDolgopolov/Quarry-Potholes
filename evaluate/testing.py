import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.utils import compute_sample_weight

from exploration.data_read import get_columns


def draw(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    test_weights = compute_sample_weight('balanced', y_test)

    weighted_mae = mean_absolute_error(y_test, y_pred, sample_weight=test_weights)

    class_order = sorted(df[target].unique())  # Get original class order
    plot_df = pd.DataFrame({
        'class': pd.Categorical(y_test, categories=class_order, ordered=True),
        'prediction': y_pred
    })

    # Create combined plot
    plt.figure(figsize=(12, 7))

    # Boxplot for distribution summary
    sns.boxplot(
        data=plot_df,
        x='class',
        y='prediction',
        hue='class',
        width=0.4,
        showfliers=True,
        zorder=1
    )
    model_name=filename.split("\\")[-1]
    sns.lineplot(x=[0,len(class_order)-1], y=[0,150], zorder=4)
    plt.title(f'{filename}\n{weighted_mae:.1f}')
    plt.tight_layout()
    # plt.savefig(f'images/{model_name}.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # target = 'class'
    # filename = 'models/GBR[lr0.1][depth4]_[ws10]-balanced24.pkl'
    # df = load_prepared(f'data/{target}10', sample_frac=0.5)
    # X, y_test = df.drop(columns=[target]), df[target]

    print(get_columns('data/class10'))
