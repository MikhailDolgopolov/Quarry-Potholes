import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

sns.set_theme(style='whitegrid', font='serif')
from matplotlib import pyplot as plt
import scienceplots
# plt.style.use(['science', 'grid', 'no-latex'])

from exploration.data_read import load_plain_data

df = load_plain_data('data/renamed')

df['vel_bin'] = pd.cut(df['vel'], bins=10)

pearson_coef, p_pearson = pearsonr(df['vel'], df['acc'])
spearman_coef, p_spearman = spearmanr(df['vel'], df['acc'])

print(f"Коэффициент Пирсона: {pearson_coef:.3f} (p-value: {p_pearson:.3e})")
print(f"Коэффициент Спирмена: {spearman_coef:.3f} (p-value: {p_spearman:.3e})")

unique_bins = sorted(df['vel_bin'].unique())
mapping = {label: np.round(label.mid) for i, label in enumerate(unique_bins)}
df['vel_bin_index'] = df['vel_bin'].map(mapping)
# print(mapping)

# Рисуем по индексам
fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(x='vel_bin_index', y='acc', data=df, showfliers=False, ax=ax)

# Подписываем метки по индексам
# print(mapping)
ax.set_xticks(np.arange(len(unique_bins)))
ax.set_xticklabels(list(mapping.values()), rotation=45)

ax.set_xlabel('Диапазон скорости')
ax.set_ylabel('Ускорение, м/с²')
plt.tight_layout()
plt.savefig('plots/boxplot_velocity_acc.png', bbox_inches='tight', dpi=300)