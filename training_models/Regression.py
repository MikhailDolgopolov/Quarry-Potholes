import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from exploration.data_read import load_preprocessed

if __name__ == "__main__":
    from sklearn.utils.class_weight import compute_sample_weight

    # Load data
    ws = 10
    target = 'class'
    df = load_preprocessed(f"data/class{ws}", sample_frac=1)

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.1)
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    # Calculate sample weights
    sample_weights = compute_sample_weight('balanced', y_train)

    # Train model with weights
    m = LinearRegression().fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate and save
    y_pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.2f}")

    with open(f'models/LinReg-[{ws}]_{round(mae)}.pkl', "wb") as f:
        pickle.dump(m, f)