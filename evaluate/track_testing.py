import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

from exploration.data_prep import current_transformer
from exploration.data_read import read_new_points
from helpers import load_pickle, discretize_to_levels, select_random_file

ran = random.randint(1, 38)
select = f'route{ran}'
track = read_new_points(select_random_file(f'data/routes/{select}'))

data = current_transformer.roll_data(track)
target='class'
model_path = "HGBR_[l2_regularization0.5][learning_rate0.6][max_iter200][min_samples_leaf5][random_state42][scoringneg_mean_absolute_error][tol0.01]_top1_21.pkl"
with open(f'models/{model_path}', 'rb') as f:
    model = pickle.load(f)

Xy = data.drop(columns=['lat', 'lon'])

X, y = Xy.drop(columns=[target]), np.clip(Xy[target], 0, 120)

pred = model.predict(X)
pred = discretize_to_levels(pred, np.arange(0, 120, 30))

plt.figure(figsize=(12, 6))

# Plot true values (y) and predictions (pred)
plt.plot(np.arange(len(pred)), y, label='True Values', color='blue', alpha=0.7)
plt.plot(np.arange(len(pred)), pred, label='Predictions', color='red', linestyle='-', alpha=0.7)
mask = (y > 0) & (pred > y)
plt.fill_between(
    np.arange(len(pred)),
    np.where(mask, pred, y),
    # where=np.where(pred>y, True, False),
    color='black',
    alpha=0.3,
    label='Concordance'
)

# Add labels and title
plt.xlabel('Seconds')
plt.ylabel('Pothole Severity')
plt.title('True Values vs Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Show plot
plt.tight_layout()
# plt.show()
plt.savefig(f'images/{select}.png')