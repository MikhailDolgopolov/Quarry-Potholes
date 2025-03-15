import glob
import pickle
import pandas as pd
from pygam import PoissonGAM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.utils import compute_sample_weight

from data_read import load_prepared

target = 'class'
filename='models/GBR[lr0.1][depth4]_[ws10]-balanced24.pkl'
df = load_prepared(f'data/{target}10', sample_frac=0.5)
X, y_test = df.drop(columns=[target]), df[target]
# with open(filename, "rb") as f:
#     model = pickle.load(f)
# y_pred = model.predict(X)

# Residual plot


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
        showfliers=False,
        zorder=1
    )
    sns.lineplot(x=[0,len(class_order)-1], y=[0,150], zorder=4)
    plt.title(f'{filename}\n{weighted_mae:.1f}')
    plt.tight_layout()
    plt.show()


for n in glob.glob('models/*.pkl'):
    draw(n)


