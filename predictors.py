import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import xgboost as xgb

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

# Step 1: Define the folder containing CSV files
folder_path = f'data/prepared5'

# Step 2: Load all CSV files into a single DataFrame
dataframes = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, sep=';', dtype=np.float32)
        df=df.drop(columns=['lat', 'lon'])
        dataframes.append(df)

main_df = pd.concat(dataframes, ignore_index=True)
train_df, test_df = train_test_split(main_df, test_size=0.3)

# print(len(train_df), len(test_df))
# Step 3: Prepare features (X) and target (y)
X_train, y_train = train_df.drop(columns=['hole']), train_df['hole']
X_test, y_test = test_df.drop(columns=['hole']), test_df['hole']

negative_count = (y_train == 0).sum()  # Assuming class 0 is the majority
positive_count = (y_train == 1).sum()  # Assuming class 1 is the minority
bias = 1.2
scale_weight = negative_count / positive_count * bias
print(f'Actual class proportion is {negative_count / positive_count:.2f}, biasing it towards potholes by {bias}: {scale_weight:.2f}')

model = xgb.XGBClassifier(scale_pos_weight=scale_weight, max_depth=6)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
