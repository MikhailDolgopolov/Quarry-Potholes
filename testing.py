import pandas as pd
from matplotlib import pyplot as plt

from data_read import load_prepared

df = load_prepared('data/prepared7')

print(df.columns)

corr_matrix = df.corr()

# Create figure and axis
plt.figure(figsize=(12, 10))
plt.matshow(corr_matrix, fignum=False)

# Add colorbar
plt.colorbar()

# Add labels
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

# Add correlation values as text
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(i, j, f'{corr_matrix.iloc[i, j]:.2f}',
                ha='center', va='center')

plt.tight_layout()
plt.show()