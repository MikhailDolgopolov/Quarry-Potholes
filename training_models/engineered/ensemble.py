import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from exploration.data_read import load_engineered_data
from helpers import train_split_by_column
from models.model_registry import predict_with_my_model
from training_models.engineered.batch_training import ModelTrainer

target = 'pothole'
cols = ModelTrainer.get_feature_sets(5)[0]
df = load_engineered_data(f'data/engineered/lerp1/rolled5 [5.0s]', keep_latlon=False, sample_frac=1)
X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.5)
# X_train, X_test = X_train[cols], X_test[cols]

train_res = predict_with_my_model(X_train, 'GLVQ')
test_res = predict_with_my_model(X_test, 'GLVQ')

m = HistGradientBoostingClassifier()
m.fit(X_train, train_res)

pred = m.predict(X_test)
print(classification_report(test_res, pred))
print(np.count_nonzero(test_res!=pred))


