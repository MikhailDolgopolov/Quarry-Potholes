import numpy as np
from sklearn.metrics import classification_report
from statsmodels.discrete.discrete_model import Logit

from exploration.data_read import load_prepared
from helpers import train_split_by_column

target, ws = 'hole', 0
df = load_prepared(f'data/{target}{ws}', keep_latlon=False, sample_frac=1)
X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.2)

model = Logit(y_train, X_train).fit()
print(model.summary())

y_pred = model.predict(X_test)
y_pred = np.where(y_pred >= 0.5, 1, 0)
print(classification_report(y_test, y_pred))