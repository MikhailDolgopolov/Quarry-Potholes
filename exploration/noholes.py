import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid

from exploration.data_read import load_prepared


def load_and_preprocess_data(target: str, sample_frac: float = 0.5) -> tuple[pd.DataFrame, pd.Series]:
    """Загрузка и предобработка данных"""
    df = load_prepared(f'data/{target}10', keep_latlon=True, sample_frac=sample_frac)
    df = df.drop(columns=['rel', 'lat', 'lon'], errors='ignore')
    return df.drop(columns=[target]), df[target]


def load_model(models_pattern: str):
    """Загрузка модели из файла"""
    model_file = glob.glob(models_pattern)[0]
    with open(model_file, "rb") as f:
        return pickle.load(f)


def prepare_clustering_data(model, X: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    """Подготовка данных для кластеризации"""
    results_df = X.copy()
    results_df['true_class'] = y_true
    results_df['prediction'] = model.predict(X)
    return results_df[results_df['true_class'] == 0].copy()


def analyze_clusters(data: pd.DataFrame, params: dict):
    """Анализ кластеров и визуализация результатов"""
    for param in tqdm(list(ParameterGrid(params)), 'searching parameter grid'):
        dbscan = DBSCAN(**param)
        clusters = dbscan.fit_predict(data)
        true_clusters = clusters[clusters != -1]
        if 1<len(np.unique(true_clusters)) <= 5 and len(true_clusters)/len(data)>0.6:
            plot_cluster_distribution(data, clusters, param)


def plot_cluster_distribution(X_zero: pd.DataFrame, clusters: np.ndarray, param: dict):
    """Визуализация распределения кластеров"""
    clustered_data = X_zero.copy()
    clustered_data['cluster'] = clusters
    clustered_data = clustered_data[clustered_data['cluster'] != -1]

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=clustered_data,
        x='prediction',
        hue='cluster',
        kde=True,
        palette='viridis'
    )
    ratio = len(clustered_data) / len(X_zero)
    plt.title(f"Params: {param}\nKept samples: {ratio:.0%}")
    plt.show()


if __name__ == '__main__':
    # Загрузка данных и модели
    X, y_test = load_and_preprocess_data('class')
    model = load_model('models/*.pkl')

    # Подготовка данных для кластеризации
    class_zero = prepare_clustering_data(model, X, y_test)
    X_zero = class_zero.drop(columns=['true_class', 'prediction'])

    # Параметры и анализ кластеров
    dbscan_params = {
        "eps": [1, 1.5, 2, 3, 4],
        "min_samples": [400, 500, 700, 900]
    }

    analyze_clusters(class_zero, dbscan_params)

